from functools import partial

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def get_tokenization_stats(example, tokenizer=None):
    messages = {
        "prompt": [
            # improves adherence to the system prompt by having it in the user context
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["text"]},
        ],
    }

    inputs = tokenizer.apply_chat_template(
        messages["prompt"], tokenize=True, add_generation_prompt=True
    )
    return {
        "input_ids": inputs,
    }


def get_dataset_lengths(dataset):
    input_ids = dataset.data.column("input_ids")
    lengths = np.vectorize(len)(np.array(input_ids, dtype=object))

    return lengths


def main():
    ds = load_dataset("skrishna/gsm8k_only_answer", split="train")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    stats = partial(get_tokenization_stats, tokenizer=tokenizer)
    ds = ds.map(stats, remove_columns=["text", "label"])
    max_input_len = np.max(get_dataset_lengths(ds))

    print(f"Max input length: {max_input_len}")

if __name__ == "__main__":
    main()