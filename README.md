# causal-relation-extraction

## Reproducing experiments locally

Run the full replication suite:

```bash
python run_all.py --suite replication --overwrite_results
```

This runs:

- main experiments on `semeval`
- main experiments on `causalnews`
- CausalNews short-span/long-span split experiments for RoBERTa and KnowPrompt

Results are written locally to:

```text
results/local_results.csv
```

W&B is disabled by default, so the experiments do not require W&B login or setup. To also log to W&B, add:

```bash
python run_all.py --suite replication --overwrite_results --use_wandb
```

Aggregate local results without W&B:

```bash
python src/models/aggregate_results.py \
  --local_file results/local_results.csv \
  --output results/local_aggregated_results.csv
```

Notes for replication:

- `causalnews` is the local dataset stored in `src/data/datasets/train_clean.txt` and `src/data/datasets/dev_clean.txt`.
- SemEval is loaded through Hugging Face datasets, so the first run needs internet access unless it is already cached.
- RoBERTa weights also need internet access on first download unless cached locally.
- The full suite trains many transformer runs and is much easier to run on a CUDA GPU.

## Docker

Build the image:

```bash
docker build -t causal-re .
```

Run the full replication suite and write results back to your local machine:

```bash
docker run --rm \
  -v "$PWD/results:/app/results" \
  causal-re
```

To reuse Hugging Face dataset/model downloads between runs:

```bash
docker run --rm \
  -v "$PWD/results:/app/results" \
  -v "$HOME/.cache/huggingface:/home/mambauser/.cache/huggingface" \
  causal-re
```

The default Docker command is equivalent to:

```bash
python run_all.py --suite replication --overwrite_results --no_wandb
```

Aggregate the local CSV inside Docker:

```bash
docker run --rm \
  -v "$PWD/results:/app/results" \
  causal-re \
  python src/models/aggregate_results.py \
    --local_file results/local_results.csv \
    --output results/local_aggregated_results.csv
```

Run one smaller experiment instead of the full suite:

```bash
docker run --rm \
  -v "$PWD/results:/app/results" \
  causal-re \
  python run_all.py --suite single --dataset causalnews --overwrite_results --no_wandb
```

GPU note:

If Docker and the NVIDIA container runtime are installed, add `--gpus all`:

```bash
docker run --rm --gpus all \
  -v "$PWD/results:/app/results" \
  causal-re
```

Without GPU access, the container can still run, but transformer experiments will be slow.
