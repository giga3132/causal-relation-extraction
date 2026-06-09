import wandb
import pandas as pd
import argparse
from collections import Counter


METRIC_KEYS = {
    "accuracy": ["eval_accuracy", "eval/accuracy", "accuracy"],
    "f1": ["eval_f1", "eval/f1", "f1"],
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=str, nargs='+', default=["causal-re-f-b"], help="List of W&B projects to fetch results from.")
    parser.add_argument("--entity", type=str, required=True) #wandb username
    parser.add_argument("--output", type=str, default="aggregated_results.csv")
    parser.add_argument("--finished-only", action="store_true", help="Only aggregate runs whose W&B state is finished.")
    args = parser.parse_args()

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
            
            acc = first_present(summary, METRIC_KEYS["accuracy"])
            f1 = first_present(summary, METRIC_KEYS["f1"])

            if acc is None or f1 is None:
                skipped_metrics += 1
                continue

            data.append({
                "project": project_name,
                "run_name": run.name,
                "state": run.state,
                "model": model,
                "dataset": dataset,
                "k": k,
                "seed": seed,
                "accuracy": float(acc),
                "f1": float(f1)
            })

    print(f"Fetched {total_runs} runs")
    print(f"Run states: {dict(states)}")
    print(f"Skipped by state: {skipped_state}")
    print(f"Skipped without accuracy/f1 metrics: {skipped_metrics}")
    print(f"Runs included: {len(data)}")

    df = pd.DataFrame(data)
    if df.empty:
        print("No results found. Please check your entity/project name, project names, run states, and metric names.")
        return

    # Group by all experimental settings (excluding seed) and aggregate
    stats = df.groupby(["model", "dataset", "k"], dropna=False).agg({
        "accuracy": ["mean", "std"],
        "f1": ["mean", "std"]
    }).reset_index()

    # Sort by model, dataset, and k for a cleaner report
    stats = stats.sort_values(["model", "dataset", "k"])

    # Flatten the multi-index columns for readability (e.g., accuracy_mean)
    stats.columns = ['_'.join(col).strip('_') for col in stats.columns.values]
    
    # Optional: Format as percentages or round
    for col in stats.columns:
        if any(x in col for x in ["mean", "std"]):
            stats[col] = stats[col].map(lambda x: f"{x:.4f}")

    print("\nAggregated Results (Mean ± Std across seeds):")
    print(stats.to_string(index=False))
    stats.to_csv(args.output, index=False)
    print(f"\nWrote {len(stats)} rows to {args.output}")

if __name__ == "__main__":
    main()
