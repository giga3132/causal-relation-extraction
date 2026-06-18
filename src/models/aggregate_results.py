import pandas as pd
import argparse
from collections import Counter


METRIC_KEYS = {
    "test": {
        "accuracy": ["eval_accuracy", "eval/accuracy", "accuracy"],
        "f1": ["eval_f1", "eval/f1", "f1"],
    },
    "short_span": {
        "accuracy": ["eval_short_span_accuracy", "eval_short_span/accuracy", "eval/short_span_accuracy"],
        "f1": ["eval_short_span_f1", "eval_short_span/f1", "eval/short_span_f1"],
    },
    "long_span": {
        "accuracy": ["eval_long_span_accuracy", "eval_long_span/accuracy", "eval/long_span_accuracy"],
        "f1": ["eval_long_span_f1", "eval_long_span/f1", "eval/long_span_f1"],
    },
}


def first_present(mapping, keys):
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def parse_run_name(run_name):
    parts = run_name.split("-")
    parsed = {"model": None, "dataset": None, "k": None, "seed": None}
    if not parts:
        return parsed

    parsed["model"] = parts[0]
    if len(parts) < 3:
        return parsed

    if parts[1].startswith("k") or parts[1] == "full":
        parsed["dataset"] = "semeval"
        shot_part = parts[1]
        seed_part = parts[2]
    elif len(parts) >= 4:
        parsed["dataset"] = parts[1]
        shot_part = parts[2]
        seed_part = parts[3]
    else:
        return parsed

    if shot_part == "full":
        parsed["k"] = -1
    elif shot_part.startswith("k"):
        try:
            parsed["k"] = int(shot_part[1:])
        except ValueError:
            pass

    if seed_part.startswith("s"):
        try:
            parsed["seed"] = int(seed_part[1:])
        except ValueError:
            pass

    return parsed


def aggregate_and_write(df, output):
    if df.empty:
        print("No results found.")
        return

    group_cols = ["model", "dataset", "k", "eval_subset"]
    if "experiment" in df.columns:
        group_cols = ["experiment", *group_cols]

    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    stats = df.groupby(group_cols, dropna=False).agg({
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"]
    }).reset_index()

    stats = stats.sort_values(group_cols)
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]

    for col in stats.columns:
        if any(x in col for x in ["mean", "std"]):
            stats[col] = stats[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")

    print("\nAggregated Results (Mean ± Std across seeds):")
    print(stats.to_string(index=False))
    stats.to_csv(output, index=False)
    print(f"\nWrote {len(stats)} rows to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=str, nargs='+', default=["causal-re-final-split"], help="List of W&B projects to fetch results from.")
    parser.add_argument("--entity", type=str) #wandb username
    parser.add_argument("--output", type=str, default="aggregated_results_split.csv")
    parser.add_argument("--finished-only", action="store_true", help="Only aggregate runs whose W&B state is finished.")
    parser.add_argument("--local_file", type=str, help="Aggregate local CSV results instead of fetching from W&B.")
    args = parser.parse_args()

    if args.local_file:
        df = pd.read_csv(args.local_file)
        aggregate_and_write(df, args.output)
        return

    if not args.entity:
        raise ValueError("--entity is required when fetching results from W&B. Use --local_file for offline aggregation.")

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required for online aggregation. Use --local_file to aggregate local results.") from exc

    api = wandb.Api()
    data = []
    total_runs = 0
    skipped_state = 0
    skipped_metrics = 0
    states = Counter()

    for project_name in args.projects:
        print(f"Fetching runs from project: {project_name}")
        runs = api.runs(f"{args.entity}/{project_name}")

        for run in runs:
            total_runs += 1
            states[run.state] += 1

            if args.finished_only and run.state != "finished":
                skipped_state += 1
                continue

            config = run.config
            summary = run.summary
            parsed_name = parse_run_name(run.name)
            
            model = config.get("model") or parsed_name["model"] or "unknown"
            dataset = config.get("dataset") or parsed_name["dataset"] or "semeval"
            k = config.get("k")
            if k is None:
                k = parsed_name["k"]
            seed = config.get("seed")
            if seed is None:
                seed = parsed_name["seed"]
            
            rows_for_run = []
            for eval_subset, metric_keys in METRIC_KEYS.items():
                acc = first_present(summary, metric_keys["accuracy"])
                f1 = first_present(summary, metric_keys["f1"])

                if acc is None or f1 is None:
                    continue

                rows_for_run.append({
                    "project": project_name,
                    "run_name": run.name,
                    "state": run.state,
                    "model": model,
                    "dataset": dataset,
                    "k": k,
                    "seed": seed,
                    "eval_subset": eval_subset,
                    "accuracy": float(acc),
                    "f1": float(f1)
                })

            if not rows_for_run:
                skipped_metrics += 1
                continue

            data.extend(rows_for_run)

    print(f"Fetched {total_runs} runs")
    print(f"Run states: {dict(states)}")
    print(f"Skipped by state: {skipped_state}")
    print(f"Skipped without accuracy/f1 metrics: {skipped_metrics}")
    print(f"Metric rows included: {len(data)}")

    df = pd.DataFrame(data)
    if df.empty:
        print("No results found. Please check your entity/project name, project names, run states, and metric names.")
        return

    aggregate_and_write(df, args.output)

if __name__ == "__main__":
    main()
