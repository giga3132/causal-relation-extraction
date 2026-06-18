import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "results" / "local_results.csv"
RESULT_FIELDS = [
    "experiment",
    "model",
    "dataset",
    "k",
    "seed",
    "eval_subset",
    "accuracy",
    "f1",
    "loss",
    "epoch",
    "train_examples",
    "test_examples",
    "lr",
    "lr1",
    "lr2",
    "max_length",
    "span_split_threshold",
    "span_split_short_n",
    "span_split_long_n",
]


def init_wandb(project, name, config, enabled=True):
    if not enabled:
        return None

    try:
        import wandb
    except ImportError:
        print("W&B is not installed; continuing with local result logging only.")
        return None

    try:
        return wandb.init(project=project, name=name, config=config)
    except Exception as exc:
        print(f"W&B initialization failed ({exc}); continuing with local result logging only.")
        return None


def wandb_log(run, metrics):
    if run is None:
        return
    try:
        run.log(metrics)
    except Exception as exc:
        print(f"W&B logging failed ({exc}); continuing.")


def wandb_finish(run):
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"W&B finish failed ({exc}); continuing.")


def append_result(output_path, row):
    path = Path(output_path) if output_path else DEFAULT_RESULTS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = {
        key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
        for key, value in row.items()
    }
    normalized = {field: normalized.get(field, "") for field in RESULT_FIELDS}

    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(normalized)

    print(f"Wrote local result to {path}")
