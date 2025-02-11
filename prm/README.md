# Process Reward Modelling with axolotl

This cookbook accompanies our [Training Process Reward Models in axolotl](<blogpost_link) blog-post, and provides details on reproducing the [axolotl-ai-co/Qwen2.5-Math-PRM-7B](https://huggingface.co/axolotl-ai-co/Qwen2.5-Math-PRM-7B) model - an open-source PRM with competitive performance.

## Datasets  

```
axolotl train prm.yaml
```

## Evaluation

To reproduce our evals, run

```bash
torchrun --nprocpernode 8 --model my_repo/my_model
```


