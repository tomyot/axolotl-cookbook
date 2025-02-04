import re

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer>, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def data_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }
    return transform_fn, {"remove_columns": ["messages"]}

def gsm8k_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["text"]},
            ],
            "solution": example["label"],
        }
    return transform_fn, {"remove_columns": ["text", "label"]}


def correct_answer_reward(completions, solution, **kwargs):
    # look for the correct answer as the last numeric match in the completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # reverse the string and search from the end for numeric matches
        matches = re.search(r"([0-9]+\$-?)|([0-9]+\-?)", content[::-1])
        reward = 0.0
        if matches:
            res = matches[0][::-1] if matches[0] else matches[1][::-1]
            if res.strip() == str(sol):
                reward = 1.0
        rewards.append(reward)
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]
