# Cerebro Reproduction

This repository reproduces the NPM side of Cerebro, based on:

> Killing Two Birds with One Stone: Malicious Package Detection in NPM and PyPI using a Single Model of Malicious Behavior Sequence

For controlled train/test experiments, use `standard_pipeline.py`. It consumes explicit split manifests, extracts behavior sequences with Jelly, fine-tunes the sequence model, runs test inference, and writes `metrics.json`.

## Environment

### 1. Node.js and Jelly

Jelly requires Node.js 22 or newer. On Windows:

```powershell
node --version
nvm use 24.8.0
node --version
npm install -g @cs-au-dk/jelly
jelly --version
```

If Jelly runs under an older Node.js, call-graph extraction may fail and `cg.json` will not be produced. Treat those results as invalid.

### 2. Python

The local environment is managed with `uv`:

```powershell
uv python install 3.14
uv sync
```

For GPU-specific PyTorch wheels, install the matching PyTorch build before running training. CPU execution works for smoke tests but can be slow.

## Inputs

Required inputs for standard evaluation:

- `--split-dir`: contains `train_manifest.json` and `test_manifest.json`.
- `--benign-train-dir` or `--benign-train-manifest`.
- `--benign-test-dir` or `--benign-test-manifest`.
- `--out-dir`: output directory.

Optional:

- `--groundtruth-jsonl`: optional annotation JSONL. Used for type/difficulty reports when available.

## Run Standard Evaluation

PowerShell:

```powershell
uv run .\standard_pipeline.py `
  --split-dir C:\path\to\split `
  --benign-train-dir C:\path\to\benign\train `
  --benign-test-dir C:\path\to\benign\test `
  --groundtruth-jsonl C:\path\to\annotations.jsonl `
  --out-dir .\outputs\standard_eval `
  --jelly-timeout 1000 `
  --parallel-batches 2 `
  --num-epochs 3 `
  --batch-size 16 `
  --max-length 256
```

Git Bash / WSL-style shell:

```bash
./run.sh standard-eval \
  --split-dir /path/to/split \
  --benign-train-dir /path/to/benign/train \
  --benign-test-dir /path/to/benign/test \
  --groundtruth-jsonl /path/to/annotations.jsonl \
  --out-dir ./outputs/standard_eval
```

Useful options:

- `--materialize`: `copy`, `hardlink`, or `symlink`.
- `--jelly-timeout`: timeout for call-graph extraction.
- `--parallel-batches`: number of extraction batches to process in parallel.
- `--process-timeout`: timeout for subprocesses inside extraction/training stages.
- `--num-epochs`, `--batch-size`, `--learning-rate`, `--max-length`: model-training controls.
- `--eval-fraction`: validation fraction from the training split.
- `--oversample`: oversample the training set.
- `--report-by-type`, `--report-by-difficulty`: write subgroup reports when ground truth includes those fields.
- `--split-stratify-by-type`, `--split-stratify-by-difficulty`: internal validation stratification options.

## Outputs

The standard output directory contains sequence files, model artifacts, predictions, and:

```text
metrics.json
```

`metrics.json` contains the binary classification metrics for the selected split.

## Manual Workflow

The lower-level scripts are still useful for debugging:

```powershell
uv run .\scripts\run_all.py --dataset_dir C:\path\to\malicious --out_dir .\outputs --jelly_timeout 1000 --label 1 --parallel_batches 6
uv run .\scripts\05_train_bert.py .\outputs\train.jsonl
uv run .\scripts\06_infer.py .\outputs\test.jsonl
```

For controlled comparisons, use `standard_pipeline.py` so train/test construction and metrics stay consistent.

## Common Issues

- `Error: Node.js >=22.0.0 is required`: switch Node.js with `nvm use 24.8.0`, then reinstall Jelly if needed.
- Missing or empty `cg.json`: Jelly failed; inspect the extraction logs under the output directory.
- CUDA/PyTorch mismatch: install a PyTorch wheel matching your CUDA runtime, or use CPU for small smoke runs.
- Long Windows paths: prefer shorter output paths such as `D:\tmp\cerebro_eval`.
