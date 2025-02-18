# Process Reward Modelling with axolotl

This cookbook accompanies our [Training Process Reward Models in axolotl](<blogpost_link) blog-post, and provides details on reproducing the evaluation results for [axolotl-ai-co/Qwen2.5-Math-PRM-7B](https://huggingface.co/axolotl-ai-co/Qwen2.5-Math-PRM-7B) model - an open-source PRM with competitive performance.

### ProcessBench

```bash
torchrun --nproc_per_node=4 eval_ProcessBench.py --model <model_path> -b 24 -w 4 -s "\n"
```

### Best of N

```bash
python bon.py --base_model <model_path> --prm_model <model_path> --num_gpus 2 --n 8
```

