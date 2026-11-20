<h1 align="center">Exponential Reward-Weighted Fine-Tuning for<br>Offline RL in Multi-Turn Conversation</h1>

<p align="center">Huy Dao &emsp; Lizi Liao</p>

<p align="center">
  <a href="https://openreview.net/forum?id=DE0tMV3YZo"><img alt="Paper on OpenReview" src="https://img.shields.io/badge/OpenReview-Paper-8c1b13"></a>
  <a href="https://neurips.cc/virtual/2026/poster/154489"><img alt="NeurIPS 2026 poster" src="https://img.shields.io/badge/NeurIPS_2026-Poster-68488f"></a>
</p>

## Overview

ExpRFT fine-tunes a dialogue model on scored conversations. It compares conversations that start from the same question and gives each assistant turn more weight when its conversation scores higher. [Reward weighting](exprft/weighting.py) · [Training loss](exprft/trainer.py)

## ⚙️ Installation

Use Python 3.11 and a GPU with BF16 support. The default training configuration fits on one 80 GB GPU. Run the blocks from the repository root in the same Bash shell.

```bash
# Data preparation, reward scoring, and training
python3.11 -m venv .venv-train
TRAIN=.venv-train/bin/python
"$TRAIN" -m pip install -r requirements.txt

# Generation and evaluation (vLLM uses a different Transformers version)
python3.11 -m venv .venv-gen
GEN=.venv-gen/bin/python
"$GEN" -m pip install vllm==0.25.1
```

Export `HF_TOKEN` with access to the base model. For GPT-4o scoring, copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`. [Request Llama 3.1 8B Instruct access](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

## 🚀 Getting Started

### 1. Prepare the MMLU questions

```bash
DS=mmlu
"$TRAIN" scripts/download_data.py --datasets "$DS"
"$TRAIN" scripts/build_contexts.py --dataset "$DS"
```

The default split uses 400 training questions and 100 test questions. For a small run, add `--n_train 20 --n_test 5` to `build_contexts.py`. [Other benchmarks](data/README.md)

### 2. Generate and score conversations

```bash
TRAJECTORIES="data/trajectories/$DS.json"
"$GEN" scripts/collect_trajectories.py --dataset "$DS" --out "$TRAJECTORIES"
"$TRAIN" scripts/compute_rewards.py --trajectories "$TRAJECTORIES"
```

The model generates three conversations per training question. GPT-4o scores each conversation and saves its reward in the same file.

### 3. Train the LoRA adapter

```bash
BETA=0.7
PROCESSED="data/processed/$DS/beta$BETA"
ADAPTER="checkpoints/$DS/beta$BETA"

"$TRAIN" scripts/build_dataset.py --dataset "$DS" --trajectories "$TRAJECTORIES" \
    --beta "$BETA" --out "$PROCESSED"
CUDA_VISIBLE_DEVICES=0 "$TRAIN" scripts/train.py --data "$PROCESSED" --out "$ADAPTER"
```

Smaller positive values of `BETA` give higher-scoring conversations more weight relative to others for the same question. [LoRA and optimizer settings](configs/train.yaml)

### 4. Evaluate on the test split

```bash
"$GEN" scripts/evaluate.py --dataset "$DS" --adapter "$ADAPTER" --out "results/$DS/test.json"
```

The JSON contains test accuracy and per-question predictions. For a base-model comparison, omit `--adapter "$ADAPTER"` and set `--out "results/$DS/test_base.json"`.

## Citation

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{dao2026exponential,
  title     = {Exponential Reward-Weighted Fine-Tuning for Offline {RL} in Multi-Turn Conversation},
  author    = {Huy Dao and Lizi Liao},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=DE0tMV3YZo}
}
```

</details>
