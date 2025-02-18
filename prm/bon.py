# credit to the HF team
# https://github.com/huggingface/search-and-learn/blob/main/src/sal/search/best_of_n.py
import argparse
import gc
import math
import os
import random
from math import perm

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


SYSTEM_PROMPT: str = (
    "Solve the following math problem efficiently and clearly:\n\n"
    "- For simple problems (2 steps or fewer):\n"
    "Provide a concise solution with minimal explanation.\n\n"
    "- For complex problems (3 steps or more):\n"
    "Use this step-by-step format:\n\n"
    "## Step 1: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "## Step 2: [Concise description]\n"
    "[Brief explanation and calculations]\n\n"
    "...\n\n"
    "Regardless of the approach, always conclude with:\n\n"
    "Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\n"
    "Where [answer] is just the final number or expression that solves the problem."
)


def find_first_zero(tensor):
    zeros = (tensor == 0).nonzero()
    return zeros[0].item() if zeros.numel() > 0 else -1


def score(
    prm_model,
    questions: list[str],
    outputs: list[list[str]],
    step_separator: str,
    tokenizer: AutoTokenizer,
) -> list[list[float]]:
    """
    This function scores a list of questions and their completions using the PRM model.
    """
    inputs_for_prm = []
    separator_ids = tokenizer.encode(step_separator, add_special_tokens=False, return_tensors="pt")
    score_idxs = []

    for question, responses in zip(questions, outputs):
        prompt_ids = tokenizer.encode(
            question + "\n",
            add_special_tokens=False,
            return_tensors="pt",
        )
        score_idxs.append([])
        for response in responses:
            steps = response.split("\n\n")
            score_idxs[-1].append([])
            for step in steps:
                step_ids = tokenizer.encode(step + "\n\n", add_special_tokens=False, return_tensors="pt")
                prompt_ids = torch.cat([prompt_ids, step_ids, separator_ids], dim=-1)
                score_idxs[-1][-1].append(prompt_ids.size(-1) - 1)
            inputs_for_prm.append(prompt_ids)

    # right pad input_ids
    pad_token_id = tokenizer.pad_token_id
    max_len = max([i.size(-1) for i in inputs_for_prm])
    for i, input_idx in enumerate(inputs_for_prm):
        inputs_for_prm[i] = torch.cat(
            [
                input_idx.squeeze(),
                torch.LongTensor([pad_token_id] * (max_len - input_idx.size(-1))),
            ]
        )
    inputs_for_prm = torch.stack(inputs_for_prm).to(torch.long).to(prm_model.device)

    with torch.no_grad():
        batch_size = 4
        all_probs = []
        for i in range(0, inputs_for_prm.size(0), batch_size):
            batch = inputs_for_prm[i : i + batch_size]
            logits = prm_model(batch).logits  # Shape: [batch, seq_len, 2]
            # Get probability of positive class (index 1)
            batch_probs = torch.softmax(logits, dim=-1)[:, :, 1].cpu()  # Shape: [batch, seq_len]
            all_probs.append(batch_probs)
            del logits
        probs = torch.cat(all_probs, dim=0)  # Combine all batches

    output_scores = []

    current_idx = 0
    for question_scores in score_idxs:
        num_completions = len(question_scores)
        question_output = []
        for i in range(num_completions):
            score_positions = question_scores[i]
            # Just get the scores at the specified positions
            score_value = probs[current_idx, score_positions].tolist()
            question_output.append(score_value)
            current_idx += 1
        output_scores.append(question_output)

    del inputs_for_prm
    torch.cuda.empty_cache()

    return output_scores


def aggregate_scores(scores: list[float], agg_strategy: str = "prod") -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def main(args):
    llm = LLM(
        model=args.base_model,
        enable_prefix_caching=True,
        seed=42,
        tensor_parallel_size=args.num_gpus,
        gpu_memory_utilization=0.3,
    )
    tokenizer = llm.get_tokenizer()

    # example problems from the MATH-5O0 dataset https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    x = {
        "problem": [
            r"Define \[p = \sum_{k = 1}^ \infty \frac{1}{k^2} \quad \text{and} \quad q = \sum_{k = 1}^\infty \frac{1}{k^3}.\]Find a way to write \[\sum_{j = 1}^\infty \sum_{k = 1}^\infty \frac{1}{(j + k)^3}\] in terms of $p$ and $q.$",
            r"A regular hexagon can be divided into six equilateral triangles. If the perimeter of one of the triangles is 21 inches, what is the perimeter, in inches, of the regular hexagon?",
            r"The expression $2\cdot 3 \cdot 4\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \[ (2\cdot (3\cdot 4)) \cdot (5+1) = 144. \]In total, how many values can be obtained from the expression $2\cdot 3\cdot 4 \cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).",
        ]
    }

    answers = ["p - q", "42", "4"]

    convs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    # tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    templated_convs = tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    templated_convs = [c for conv in templated_convs for c in [conv] * args.n]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=2048,
        top_p=1.0,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    responses = llm.generate(
        templated_convs,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    if len(responses) != len(x["problem"]) * args.n:
        raise ValueError(f"Generated {len(responses)} responses instead of {len(x['problem'] * args.n)}")

    for i in range(len(completions)):
        completions[i] = [output.text for r in responses[i * args.n : (i + 1) * args.n] for output in r.outputs]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != args.n:
            raise ValueError(f"Generated {len(c)} completions instead of {args.n}")

    # destroy vllm process
    destroy_model_parallel()
    destroy_distributed_environment()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    
    # completions is now a List[List[str]] of size NUM_PROMPTS [N]

    if args.n > 1:
        with torch.device("cuda"):
            prm_model = AutoModelForTokenClassification.from_pretrained(args.prm_model).to(torch.bfloat16)

        tokenizer = AutoTokenizer.from_pretrained(args.prm_model)

        prm_model.eval()

        scores = score(prm_model, x["problem"], completions, args.separator, tokenizer)
        agg_scores = [[aggregate_scores(s, agg_strategy="prod") for s in score] for score in scores]

        # Select the completion with the highest score
        pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]
    else:
        pred = [c[0] for c in completions]

    with open("bon_results.txt", "w") as f:
        for i in range(len(x["problem"])):
            separator = "=" * 80
            print(separator)
            print("Problem: ", x["problem"][i])
            print("Predicted answer (BoN): ", pred[i])
            print("Correct Answer: ", answers[i])

            # Write to file
            f.write(f"{separator}\n")
            f.write(f"Problem: {x['problem'][i]}\n")
            f.write(f"Predicted answer (BoN): {pred[i]}\n")
            f.write(f"Correct Answer: {answers[i]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--prm_model", type=str)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n",
        help="It's important to use the same separator as the one used during TRL training",
    )
    parser.add_argument("--n", type=int, default=8)

    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
