# Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination in LVLMs (EMNLP'25 Oral)

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)
[![Arxiv](https://img.shields.io/badge/arXiv-2411.09968-B21A1B)](https://arxiv.org/abs/2411.09968)

This repository provides the official PyTorch implementation of the following paper: 
> [**Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination in LVLMs**](https://arxiv.org/abs/2411.09968) <br>

## Overview

<p align="center"><img src="./teaser.png" alt="teaser" width="500px" /></p>

Hallucination, posed as a pervasive challenge of multi-modal large language models (MLLMs),  has significantly impeded their real-world usage that demands precise judgment. Existing  methods  mitigate  this  issue  with  either training with specific designed data or inferencing with external knowledge from other sources,  incurring inevitable additional costs. In  this  paper,  we  present OPERA,  a novel MLLM decoding method grounded in an Over-trust Penalty and a Retrospection-Allocation strategy, serving as a nearly free lunch to alleviate the hallucination issue without additional data, knowledge, or training.  Our approach begins with an  interesting observation that,  most  hallucinations are closely tied to the knowledge aggregation patterns manifested in the self-attention matrix, i.e.,  MLLMs tend to generate new tokens by focusing on a few summary tokens, but not all the previous tokens.  Such partial over-trust  inclination  results  in  the  neglecting  of  image  tokens and describes  the image content  with  hallucination.   Statistically,  we  observe  an  80%∼95%  co-currency  rate  between hallucination contents and such knowledge aggregation patterns. Based on the observation, OPERA introduces a penalty term on the model logits during the beam-search decoding to mitigate the over-trust issue, along with a rollback strategy that retrospects the presence of summary tokens in the previously generated tokens, and re-allocate the token  selection  if  necessary.   With  extensive  experiments, OPERA shows significant hallucination-mitigating performance on different MLLMs and metrics, proving its effec-tiveness and generality. 


## Setup

The main implementation of EAH is in `transformers-4.29.2/src/transformers/models/llama/modeling_llama`.

So it is convenient to use EAH decoding by just changing original `modeling_llama` to our `modeling_llama_eah`.
```
conda env create -f environment.yml
conda activate eah
python -m pip install -e transformers-4.29.2
```
#### Note: to implement EAH on other version of transformers, you can follow the steps as the follows:
- Find the filefolder  `transformers-4.29.2/src/transformers/models/llama/`.
- Overwrite original `modeling_llama` to our `modeling_llama_eah` (with file name `modeling_llama`).




## Evaluation

The following evaluation requires for MSCOCO 2014 dataset. Please download [here](https://cocodataset.org/#home) and extract it in your data path.

Besides, it needs you to prepare the following checkpoints of 7B base models:

- Download [LLaVA-1.5 merged 7B model](https://huggingface.co/liuhaotian/llava-v1.5-7b) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/llava-1.5_eval.yaml#L14) of `eval_configs/llava-1.5_eval.yaml`.
- Download [Vicuna 7B v1.1 model](https://github.com/lm-sys/FastChat) and specify it at [Line 25](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/blip2_instruct_vicuna7b.yaml#L25) of `minigpt4/configs/models/blip2_instruct_vicuna7b.yaml`.
- Download [Vicuna 7B v0 model](https://huggingface.co/Vision-CAIR/vicuna-7b/tree/main) and specify it at [Line 18](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/minigpt4/configs/models/minigpt4_vicuna0.yaml#L18) of `minigpt4/configs/models/minigpt4_vicuna0.yaml`.
- Download [MiniGPT-4 7B pretrained weights](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) and specify it at [Line 8](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/minigpt4_eval.yaml#L8) of `eval_configs/minigpt4_eval.yaml`.
- Download [Shikra merged 7B model](https://github.com/shikras/shikra#checkpoint) and specify it at [Line 14](https://github.com/shikiw/OPERA/blob/bf18aa9c409f28b31168b0f71ebf8457ae8063d5/eval_configs/shikra_eval.yaml#L14) of `eval_configs/shikra_eval.yaml`.

### Arguments

| Argument             | Example             | Description   |
| -------------------- | ------------------- | ------------- |
| `--model`    | `llava-1.5` | Specify the MLLM model, this codebase supports `instructblip`, `minigpt4`, `llava-1.5`, `shikra`. |
| `--data-path`     | `/path/to/dataset` | Path to the dataset file or folder, e.g., `COCO_2014/val2014/`. |
| `--pope-type`     | `random` | Type for POPE evaluation, supports `random`, `popular`, `adversarial`. |
| `--LAYER_NUM`   | `1` | Specifies which Transformer layer’s attention will be processed.. Default: 1. |
| `--HEAD_NUM`      | `k` | Specifies which attention head / head index is targeted for inspection or broadcasting. Default: k. |
| `--THRES`   | `0.002` | Sets the attention strength threshold used to identify significant image-token attention.  |


### POPE
```bash
python pope_eval.py --model MODEL_NAME --data_path /path/to/COCO --pope-type random --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```

### CHAIR
- Generate the MLLM's responses and save them in a jsonl file:
```bash
python chair_eval.py --model MODEL_NAME --data_path /path/to/COCO --gpu-id GPU_IDs --beam 5 --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```
Note: Please check out our released results in `log/chair_eval_results` for reproduction.

- Calculate CHAIR using the generated jsonl file:
```bash
python chair.py --cap_file /path/to/jsonl --image_id_key image_id --caption_key caption --coco_path /path/to/COCO/annotations_trainval2014/annotations/ --save_path /path/to/save/jsonl
```

### GPT-4V
The GPT-4V evaluation requires you to specify your API key in [Line 88](https://github.com/shikiw/OPERA/blob/559556048224d5c3eae995a21d529156fb150d5f/gpt4v_eval.py#L88) of `gpt4v_eval.py`.
```bash
python gpt4v_eval.py --model MODEL_NAME --data_path /path/to/COCO --gpu-id GPU_IDs --scale_factor 50 --threshold 15 --num_attn_candidates 5 --penalty_weights 1
```




## Acknowledgement
This repo is based on the MLLM codebase of [OPERA](https://github.com/shikiw/OPERA), [LAVIS](https://github.com/salesforce/LAVIS) and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) and the CHAIR code of [Maxlinn](https://github.com/Maxlinn/CHAIR-metric-standalone). Thanks for their impressive works!



## Citation
```bibtex
@article{seeing,
  title={Seeing clearly by layer two: Enhancing attention heads to alleviate hallucination in lvlms},
  author={Zhang, Xiaofeng and Quan, Yihao and Gu, Chaochen and Shen, Chen and Yuan, Xiaosong and Yan, Shaotian and Cheng, Hao and Wu, Kaijie and Ye, Jieping},
  journal={The 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}


@inproceedings{shallow,
  title={Shallow Focus, Deep Fixes: Enhancing Shallow Layers Vision Attention Sinks to Alleviate Hallucination in LVLMs},
  author={Zhang, Xiaofeng and Quan, Yihao and Shen, Chen and Gu, Chaochen and Yuan, Xiaosong and Yan, Shaotian and Cao, Jiawei and Cheng, Hao and Wu, Kaijie and Ye, Jieping},
  booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}
```
