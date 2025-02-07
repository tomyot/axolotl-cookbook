# adapted from https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
import re

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""


def axo_gsm8k_transform(cfg, *args, **kwargs):
    # skrishna/gsm8k_only_answer dataset

    def transform_fn(example, tokenizer=None):
        label = example["label"].replace(",", "")  # remove commas, e.g. thousands separators

        return {
            "prompt": [
                # improves adherence to the system prompt by having it in the user context
                {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + example["text"]},
            ],
            "answer": label,
        }

    return transform_fn, {"remove_columns": ["text", "label"]}

def extract_xml_answer(text: str) -> str:
    # collect the answer between the last <answer></answer> tag
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """
    gsm8k answers are ints, so rewarding for ints should help steer the model performance
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\s?$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function that loosely checks if the completion has a specific format,
    without penalizing adherence to newlines.
    """
    pattern = r"<reasoning>.*?</reasoning>\s?<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.S) for r in responses]
    return [0.25 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>") == 1:
        count += 0.125
    if text.count("</reasoning>") == 1:
        count += 0.125
    if text.count("<answer>") == 1:
        count += 0.125
    if text.count("</answer>") == 1:
        count += 0.125
        # penalize extra tokens after the answer tag
        count -= (len(text.split("</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward function for having exactly one of each <reasoning>, </reasoning>, <answer>, and </answer> tag.
    """
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]