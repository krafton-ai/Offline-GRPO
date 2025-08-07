# Continual Post-Training of LLMs via Offline GRPO for Mathematical Reasoning

A continual post-training method for enhancing state-of-the-art (SOTA) LLMs for mathematical reasoning using Offline Group Relative Policy Optimization (GRPO).

## Introduction

We propose a continual post-training method that can be applied to various reasoning-focused Large Language Models (LLMs) to further improve their performance. Our approach utilizes Offline Reinforcement Learning with Verifiable Rewards (Offline RLVR) to overcome the limitations of traditional on-policy methods.
For detailed methodology and experimental results, please refer to our blog post:
- Blog Post: Continual Post-Training of LLMs via Offline GRPO for Mathematical Reasoning [[KR](https://krafton-ai.github.io/blog/llm_post_training_kr/) / [EN](https://krafton-ai.github.io/blog/llm_post_training_en/)]

### Why Offline RLVR?

Traditional on-policy RLVR methods have two main limitations:
1. **Slow training speed**: Requires generating rollouts for each problem at every training step
2. **Performance bottleneck**: Limited by the base LLM's initial capability to generate correct trajectories

Our offline approach addresses these issues by leveraging pre-generated high-quality reasoning trajectories from teacher models, enabling faster training and better performance improvements.

### Key Features:
- **Offline RL**: Utilizes teacher model rollout trajectories for more efficient training
- **Enhanced GRPO**: Addresses the challenge of all-positive reasoning traces with bias term addition
- **Improved Performance**: Demonstrates superior results compared to standard SFT and on-policy methods

---

## Installation

You can install dependencies by running the following commands:
```bash
conda create -n offline-grpo python=3.10
conda activate offline-grpo
cd src
pip install -r requirements.txt
pip install -e .
cd verl
pip install -e .
```

If you encounter issues when installing flash-attn, we recommend you to install it here
[flash-attn](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.3). For example, we use this version.
```bash
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

## Repo Structure

This repository includes:

- `src`: Codes for training using off-policy reasoning traces. Our main code changes are in src/verl/verl/mix_src.
- `data`: Data and code for training and evaluating our method.
- `exp_scripts`: Example script to train with offline GRPO.
- `eval_scripts`: Evaluation scripts on math and out-of-distribution benchmarks. We use [SkyThought](https://github.com/NovaSky-AI/SkyThought) for evaluation here.

Our implementation is built on top of the verl framework and supports plug-and-play integration with off-policy traces from models such as DeepSeek-R1.

---

## Usage

### Data Download
We provide pre-processed datasets for training:
- Download the OpenThought3 filtered math dataset: [Link will be provided]
- The dataset contains positive and negative reasoning trajectories from teacher models

### Training

We provide an example script for training with offline GRPO on the prepared data:

```bash
cd exp_scripts
bash train_openthinker3.sh
```
This script launches multi-GPU training for the target model using our offline GRPO method.

---

## Credits

This project heavily builds upon the excellent [LUFFY](https://github.com/ElliottYan/LUFFY) repository. We extend our sincere gratitude to the original LUFFY authors for providing:

- **Core Framework**: The foundational reinforcement learning framework
- **GRPO Implementation**: The Group Relative Policy Optimization algorithm implementation
- **Training Infrastructure**: Multi-GPU training setup and data processing pipelines

Our contributions include modifications to handle only off-policy reasoning traces and enhanced data processing for continual post-training scenarios. The majority of the codebase, especially the core training loop and infrastructure, comes from the original LUFFY implementation.

We also acknowledge the use of:
- [veRL](https://github.com/volcengine/verl) for reinforcement learning infrastructure
- [deepscaler](https://github.com/agentica-project/rllm) for scaling utilities
- [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- [Math-Verify](https://github.com/huggingface/Math-Verify) for math reasoning evaluation

---

## Citation

```
@inproceedings{krafton2025encontinualposttrainingof,
  author = {KRAFTON,  and SKT, },
  title = {[EN] Continual Post-Training of LLMs via Offline GRPO for Mathematical Reasoning},
  abstract = {In this post, we explore a new approach to enhancing the reasoning capabilities of LLMs through continual post-training. While pre-training equips LLMs with broad linguistic knowledge, it often falls short in complex reasoning tasks like math or code. Recent models have shown that Reinforcement Learning with Verifiable Rewards (RLVR) can help bridge this gap, but existing methods rely on slow and limited online training. We propose an offline alternative using teacher-generated trajectories and introduce a novel variant of Group Relative Policy Optimization (GRPO) that better captures high-quality reasoning traces—even when all outputs are positive. Our experiments on mathematical reasoning show that this method leads to consistent improvements.},
  year = {2025},
  date = {July 28, 2025},
  url  = {https://krafton-ai.github.io/blog/llm_post_training_en/}
}
```
