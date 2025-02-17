#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import math
import os
import random
from math import perm

import numpy as np
import torch
from accelerate import Accelerator
from transformers import AutoModelForTokenClassification, AutoTokenizer

# from sal.config import Config
# from sal.models.reward_models import PRM
# from sal.utils.score import aggregate_scores
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


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

CUSTOM_CHAT_TEMPLATE: str = (
    "{%- if custom_tools is defined %}\n"
    "    {%- set tools = custom_tools %}\n"
    "{%- endif %}\n"
    "{%- if not tools_in_user_message is defined %}\n"
    "    {%- set tools_in_user_message = true %}\n"
    "{%- endif %}\n"
    "{%- if not date_string is defined %}\n"
    "    {%- if strftime_now is defined %}\n"
    '        {%- set date_string = strftime_now("%d %b %Y") %}\n'
    "    {%- else %}\n"
    '        {%- set date_string = "26 Jul 2024" %}\n'
    "    {%- endif %}\n"
    "{%- endif %}\n"
    "{%- if not tools is defined %}\n"
    "    {%- set tools = none %}\n"
    "{%- endif %}\n\n"
    "{#- This block extracts the system message, so we can slot it into the right place. #}\n"
    "{%- if messages[0]['role'] == 'system' %}\n"
    "    {%- set system_message = messages[0]['content']|trim %}\n"
    "    {%- set messages = messages[1:] %}\n"
    "{%- else %}\n"
    '    {%- set system_message = "" %}\n'
    "{%- endif %}\n\n"
    "{#- System message #}\n"
    '{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n'
    "{%- if tools is not none %}\n"
    '    {{- "Environment: ipython\\n" }}\n'
    "{%- endif %}\n"
    '{{- "Cutting Knowledge Date: December 2023\\n" }}\n'
    '{{- "Today Date: " + date_string + "\\n\\n" }}\n'
    "{%- if tools is not none and not tools_in_user_message %}\n"
    '    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n'
    '    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n'
    '    {{- "Do not use variables.\\n\\n" }}\n'
    "    {%- for t in tools %}\n"
    "        {{- t | tojson(indent=4) }}\n"
    '        {{- "\\n\\n" }}\n'
    "    {%- endfor %}\n"
    "{%- endif %}\n"
    "{{- system_message }}\n"
    '{{- "<|eot_id|>" }}\n\n'
    "{#- Custom tools are passed in a user message with some extra guidance #}\n"
    "{%- if tools_in_user_message and not tools is none %}\n"
    "    {#- Extract the first user message so we can plug it in here #}\n"
    "    {%- if messages | length != 0 %}\n"
    "        {%- set first_user_message = messages[0]['content']|trim %}\n"
    "        {%- set messages = messages[1:] %}\n"
    "    {%- else %}\n"
    '        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n'
    "{%- endif %}\n"
    "    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n"
    '    {{- "Given the following functions, please respond with a JSON for a function call " }}\n'
    '    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n'
    '    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n'
    '    {{- "Do not use variables.\\n\\n" }}\n'
    "    {%- for t in tools %}\n"
    "        {{- t | tojson(indent=4) }}\n"
    '        {{- "\\n\\n" }}\n'
    "    {%- endfor %}\n"
    '    {{- first_user_message + "<|eot_id|>"}}\n'
    "{%- endif %}\n\n"
    "{%- for message in messages %}\n"
    "    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n"
    "        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] + '<|eot_id|>' }}\n"
    "    {%- elif 'tool_calls' in message %}\n"
    "        {%- if not message.tool_calls|length == 1 %}\n"
    '            {{- raise_exception("This model only supports single tool-calls at once!") }}\n'
    "        {%- endif %}\n"
    "        {%- set tool_call = message.tool_calls[0].function %}\n"
    "        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n"
    "        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n"
    "        {{- '\"parameters\": ' }}\n"
    "        {{- tool_call.arguments | tojson }}\n"
    '        {{- "}" }}\n'
    '        {{- "<|eot_id|>" }}\n'
    '    {%- elif message.role == "tool" or message.role == "ipython" %}\n'
    '        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n'
    "        {%- if message.content is mapping or message.content is iterable %}\n"
    "            {{- message.content | tojson }}\n"
    "        {%- else %}\n"
    "            {{- message.content }}\n"
    "        {%- endif %}\n"
    '        {{- "<|eot_id|>" }}\n'
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- if add_generation_prompt %}\n"
    "    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n"
    "{%- endif %}\n"
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

    Args:
        questions: A list of B questions to score.
        outputs: A list of lists of completions to score:
            B questions, and N completions per question (for BoN)

    Returns:
        A list of lists of scores for each completion, i.e. B lists of N scores for each completion.
    """

    inputs_for_prm = []
    separator_ids = tokenizer.encode(
        step_separator, add_special_tokens=False, return_tensors="pt"
    )
    score_idxs = []
    for question, responses in zip(questions, outputs):
        prompt_ids = tokenizer.encode(
            SYSTEM_PROMPT + "\n" + question + "\n",
            add_special_tokens=False,
            return_tensors="pt",
        )
        score_idxs.append([])
        for response in responses:
            steps = response.split("\n")
            score_idxs[-1].append([])
            for step in steps:
                step_ids = tokenizer.encode(
                    step, add_special_tokens=False, return_tensors="pt"
                )
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
    inputs_for_prm = torch.stack(inputs_for_prm)
    inputs_for_prm = inputs_for_prm.to(prm_model.device).to(torch.long)

    with torch.no_grad():
        logits = prm_model(inputs_for_prm).logits  # Shape: [batch, seq_len, 2]
        # Get probability of positive class (index 1)
        probs = torch.softmax(logits, dim=-1)[:, :, 1]  # Shape: [batch, seq_len]

    output_scores = []
    import pdb

    pdb.set_trace()
    # for i, score_idx in enumerate(score_idxs):
    # output_scores.append(logits[i, score_idx]])

    # # TODO: tokenize each batch independently so there is less padding and faster inference
    # output_scores = batched_math_shepherd_inference(
    #     self.model,
    #     self.tokenizer,
    #     inputs_for_prm,
    #     self.search_config.prm_batch_size,
    # )
    # cumulative_lengths = list(accumulate(lengths))
    # # reshape the output scores to match the input
    # output_scores = [
    #     output_scores[i:j]
    #     for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
    # ]

    # # stripped_output_scores = [] TODO: strip out the reward for previous steps
    # for output_score, output in zip(output_scores, outputs):
    #     assert len(output_score) == len(output), f"{len(output_score)} != {len(output)}"
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

    pdb.set_trace()
    return output_scores


def aggregate_scores(scores: list[float], agg_strategy) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def main(args):
    # llm = LLM(
    #     model=args.base_model,
    #     enable_prefix_caching=True,
    #     seed=42,
    #     tensor_parallel_size=args.num_gpus,
    # )
    # tokenizer = llm.get_tokenizer()

    # generate some dummy data
    x = {
        "problem": [
            "What is the sum of 1 and 2?",
            "What is the product of 3 and 4?",
        ]
    }

    convs = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        for prompt in x["problem"]
    ]
    # tokenizer.chat_template = CUSTOM_CHAT_TEMPLATE
    # templated_convs = tokenizer.apply_chat_template(
    #     convs, tokenize=False, add_generation_prompt=True
    # )

    # Duplicate convs to generate config.n completions per prompt so we can do continous batching
    # This makes [p1, p2, p3, p4] become [p1, p1, p2, p2, p3, p3, p4, p4] for e.g. config.n=2
    # templated_convs = [c for conv in templated_convs for c in [conv] * args.n]

    # Initialize empty lists for completions and completion tokens
    completions = [[] for _ in range(len(x["problem"]))]
    completion_tokens = [[] for _ in range(len(x["problem"]))]

    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=2048,
        top_p=1.0,
        n=1,  # Since we've already duplicated the prompt_token_ids, we only need to generate 1 completion per prompt
    )

    # responses = llm.generate(
    #     templated_convs,
    #     sampling_params=sampling_params,
    #     use_tqdm=False,
    # )
    # Create simple dummy responses
    class DummyOutput:
        def __init__(self, text):
            self.text = text
            
    class DummyResponse:
        def __init__(self, text):
            self.outputs = [DummyOutput(text)]
            
    responses = []
    for i in range(len(x["problem"])):
        for j in range(args.n):
            if i == 0:
                # Responses for "What is the sum of 1 and 2?"
                if j == 0:
                    # Add a deliberately bad response 
                    responses.append(DummyResponse("Step 1: First, I'll count from 1 to 2 on my fingers. I see that I'm using 2 fingers, so that's good.\nStep 2: Now I'll add them together\nIf I count all my raised fingers, I get 4.\n Therefore, 1 + 2 = 4"))
                else:
                    responses.append(DummyResponse("Step 1: To add 1 and 2, I'll start with 1. I have 1 as my base number.\nStep 2: Then add 2 to it\nAdding 2 to 1 gives us 3.\n Therefore, 1 + 2 = 3"))
            else:
                # Responses for "What is the product of 3 and 4?"
                responses.append(DummyResponse("Step 1: To multiply 3 and 4, I'll write out 3 four times. 3 + 3 + 3 + 3\nStep 2: Now add these numbers. 3 + 3 + 3 + 3 = 12.\n Therefore, 3 × 4 = 12"))


    if len(responses) != len(x["problem"]) * args.n:
        raise ValueError(
            f"Generated {len(responses)} responses instead of {len(x['problem'] * args.n)}"
        )

    for i in range(len(completions)):
        completions[i] = [
            output.text
            for r in responses[i * args.n : (i + 1) * args.n]
            for output in r.outputs
        ]
        # completion_tokens[i] = [
        #     len(output.token_ids)
        #     for r in responses[i * args.n : (i + 1) * args.n]
        #     for output in r.outputs
        # ]

    # Check we generated the correct number of completions for each prompt
    for c in completions:
        if len(c) != args.n:
            raise ValueError(f"Generated {len(c)} completions instead of {args.n}")

    # completions is now a List[List[str]] of size NUM_PROMPTS [N]

    prm_model = AutoModelForTokenClassification.from_pretrained(args.prm_model)
    tokenizer = AutoTokenizer.from_pretrained(args.prm_model)

    prm_model.eval()

    scores = score(prm_model, x["problem"], completions, args.separator, tokenizer)
    # print("scores", scores, scores.shape)

    agg_scores = [[aggregate_scores(s, "prod") for s in score] for score in scores]

    # Select the completion with the highest score
    pred = [completion[np.argmax(s)] for completion, s in zip(completions, agg_scores)]

    print("pred", pred)
    # x["completions"] = completions
    # x["scores"] = scores
    # x["pred"] = pred
    # x["completion_tokens"] = completion_tokens

    # return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--prm_model", type=str)
    # parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n",
        help="It's important to use the same separator as the one used during TRL training",
    )
    parser.add_argument("--n", type=int, default=2)

    args = parser.parse_args()

    set_seed(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main(args)
