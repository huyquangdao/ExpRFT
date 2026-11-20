# Data

Expected layout after running the pipeline:

```
data/
  raw/<dataset>/            raw benchmark files (you download these)
  contexts/<dataset>/       train.json, test.json, splits.json (train / test ids)
  trajectories/<dataset>.json   base-policy conversations + judge rewards
  processed/<dataset>/<run>/    tokenised, weighted training examples
checkpoints/<dataset>/<run>/    LoRA adapters written by scripts/train.py
```

All commands are run from the repository root.  `<dataset>` is one of
`openbookqa | arc | sciqa | mmlu | cosql | mathdial`.

## 1. Download the raw data yourself

The raw benchmarks are **not** included.  Five of the six are public Hugging Face
datasets and are fetched into `data/raw/<dataset>/data.jsonl` by:

```bash
python scripts/download_data.py --data_dir data                      # all five HF datasets
python scripts/download_data.py --data_dir data --datasets mmlu arc   # a subset
```

| dataset    | source (Hugging Face)      | config(s)                | splits used              |
|------------|----------------------------|--------------------------|--------------------------|
| openbookqa | `allenai/openbookqa`       | `main`                   | train + validation + test|
| arc        | `allenai/ai2_arc`          | `ARC-Easy`, `ARC-Challenge` | train + validation + test|
| sciqa      | `derek-thomas/ScienceQA`   | –                        | train + validation + test|
| mmlu       | `cais/mmlu`                | `all`                    | test + validation        |
| mathdial   | `eth-nlped/mathdial`       | –                        | train + test             |

**CoSQL** has no Hugging Face mirror and must be downloaded manually from
<https://yale-lily.github.io/cosql>.  Unpack the archive so that these two files
exist:

```
data/raw/cosql/sql_state_tracking/cosql_train.json
data/raw/cosql/cosql_dataset/tables.json
```

Please respect each benchmark's own licence and terms of use; this repository does
not redistribute any of them.

## 2. Build contexts (train / test splits)

Samples 400 unique train questions and 100 unique test questions (de-duplicated, so
train and test are disjoint).

```bash
python scripts/build_contexts.py --dataset <dataset> --data_dir data --out data/contexts
# or every supported dataset at once:
python scripts/build_contexts.py --dataset all --data_dir data --out data/contexts
```

Writes `data/contexts/<dataset>/{train.json,test.json,splits.json}`.
Options: `--n_train 400 --n_test 100 --seed 42`.

**CoSQL and MathDial only.**  These benchmarks have no multiple-choice options, so
distractors are written by GPT-4o (needs `OPENAI_API_KEY`, see the top-level
README):

```bash
python scripts/make_hard_distractors.py --dataset cosql    --contexts data/contexts
python scripts/make_hard_distractors.py --dataset mathdial --contexts data/contexts
```

## 3. Collect trajectories from the base policy

Runs three scripted 3-turn conversations per train context (sampling temperatures
0.3 / 0.7 / 1.0) with `meta-llama/Llama-3.1-8B-Instruct` under vLLM.  Requires a GPU
and the vLLM environment.

```bash
python scripts/collect_trajectories.py --dataset <dataset> --contexts data/contexts \
    --out data/trajectories/<dataset>.json
```

Useful options: `--temperatures 0.3 0.7 1.0`, `--num_turns 3`,
`--max_new_tokens 1024`, `--tensor_parallel_size 1`, `--seed 0`,
`--limit N` (first N contexts only).

## 4. Judge rewards

Scores every trajectory with GPT-4o (needs `OPENAI_API_KEY`) and writes the rewards
back into the same trajectory file (`reward = Overall / 10`; the raw judge text is
kept).

```bash
python scripts/compute_rewards.py --trajectories data/trajectories/<dataset>.json
# options: --judge_model gpt-4o --max_concurrent 30
```

## 5. Build the processed training set

Applies the ExpRFT weighting `w_i = exp(r̃_i / β)` with a fixed β and tokenises one
example per assistant turn.  All 400 train contexts are used.

```bash
python scripts/build_dataset.py --trajectories data/trajectories/<dataset>.json \
    --contexts data/contexts --dataset <dataset> --beta 0.5 \
    --out data/processed/<dataset>/beta0.5
```

Options: `--sigma_floor`, `--max_seq_length 4096`, `--keep_truncated`,
`--model` (tokenizer source).
