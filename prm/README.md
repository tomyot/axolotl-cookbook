# Process Reward Modelling with axolotl

This cookbook accompanies our [Training Process Reward Models in axolotl](<blogpost_link) blog-post, and provides details on reproducing the evaluation results for [axolotl-ai-co/Qwen2.5-Math-PRM-7B](https://huggingface.co/axolotl-ai-co/Qwen2.5-Math-PRM-7B)/

### ProcessBench

```bash
torchrun --nproc_per_node=4 eval_process_bench.py --model axolotl-ai-co/Qwen2.5-Math-PRM-7B -b 24 -w 4 -s "\n"

GSM8K:
err   corr   F1
----- ------ ----
55.5   98.4  71.0

MATH:
err   corr   F1
----- ------ ----
49.8   91.9  64.6

OlympiadBench:
err   corr   F1
----- ------ ----
31.2   87.3  46.0

Omni-MATH:  
err   corr   F1
----- ------ ----
24.6   87.1  38.3

Average F1 across datasets: 55.0
```

### Best of N

```bash
python bon.py --base_model Qwen/Qwen2.5-1.5B-Instruct  --prm_model axolotl-ai-co/Qwen2.5-Math-PRM-7B --n 16
```

Example outputs can be seen in `bon_qwen1.5B-instruct_n=16_results.txt`, and `bon_qwen1.5B-instruct_n=1_results.txt`, for `n=16` and `n=1` respectively.
