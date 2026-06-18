# ReNIO: Reweighting Negative Trajectory Importance for LLM On-Policy Distillation

This repository contains the implementation of **ReNIO**.

## Installation

```bash
conda env create -f environment.yml
conda activate opsd
```

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```

## Data

For Math task, the data can be download from [here](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd).
For Coding task, the data can be download from [here](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k), we sample 30k code domain data from it.

Please put the training data in data/.

## Training

We provide the training shells in scripts/, change the `model_name_or_path` to your real model path to use them.

### GRPO

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

### OPD

See [`scripts/run_opd_1b.sh`](scripts/run_opd_1b.sh).

### OPSD

See 

[`scripts/run_opsd_1b.sh`](scripts/run_opsd_1b.sh).
[`scripts/run_opsd_4b.sh`](scripts/run_opsd_4b.sh).
[`scripts/run_opsd_8b.sh`](scripts/run_opsd_8b.sh).

To use renio, you can try
```
CLIP=2.5 \
IMP=0.8 \
RENIO=True \
bash scripts/run_opsd_1b.sh
```
for math task OPSD training on qwen3-1.7B. And use

```
DATA="data/openthoughts/openthoughts_coding_30k.jsonl" \
TASK="coding" \
CLIP=2.5 \
IMP=0.8 \
RENIO=True \
bash scripts/run_opsd_1b.sh
```
for coding task training.

Here `RENIO=True` enables ReNIO for training, `CLIP` and `IMP` controls the student-teacher log ratio clip range and the threshold for key token selection.

## Evaluation

### Math Task

See [`eval\run_eval.sh`](eval\run_eval.sh).

### Coding Task

See [`eval\run_eval_code.sh`](eval\run_eval_code.sh)

## Acknowledgements

Our implementation builds on [OPSD](https://github.com/siyan-zhao/OPSD).
