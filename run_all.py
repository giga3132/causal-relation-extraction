import subprocess
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval", help="Dataset to use in single-suite mode: semeval or causalnews.")
parser.add_argument("--suite", choices=["single", "replication"], default="single", help="single runs one dataset; replication runs SemEval main, CausalNews main, and CausalNews span split experiments.")
parser.add_argument("--span_split_eval", action="store_true", help="Evaluate RoBERTa/KnowPrompt on short vs long entity-span test subsets.")
parser.add_argument("--span_split_threshold", type=float, default=None, help="Short/long threshold over max entity span length.")
parser.add_argument("--results_file", type=str, default="results/local_results.csv")
parser.add_argument("--overwrite_results", action="store_true", help="Delete the local results file before running.")
parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging. Local CSV logging still runs.")
parser.add_argument("--use_wandb", action="store_true", help="Also log final metrics to W&B if it is installed and configured.")

k_values = [8, 16, 32, 64, 128, 256, -1]
seeds = [42, 43, 44, 45, 46]
models = [
    "src/models/baseline.py",
    "src/models/roberta.py",
    "src/models/knowprompt.py"
]

ROBERTA_LR_BY_K = {
    8: 1e-5,
    16: 2e-5,
    32: 3e-5,
    64: 3e-5,
    128: 5e-5,
    256: 5e-5,
    -1: 5e-5,
}

KNOWPROMPT_LR_BY_K = {
    8: {"lr1": 1e-4, "lr2": 1e-5},
    16: {"lr1": 1e-4, "lr2": 2e-5},
    32: {"lr1": 1e-4, "lr2": 3e-5},
    64: {"lr1": 2e-4, "lr2": 3e-5},
    128: {"lr1": 2e-4, "lr2": 5e-5},
    256: {"lr1": 2e-4, "lr2": 5e-5},
    -1: {"lr1": 2e-4, "lr2": 5e-5},
}


def learning_rate_args(model_path, k):
    if model_path.endswith("roberta.py"):
        return ["--lr", str(ROBERTA_LR_BY_K[k])]

    if model_path.endswith("knowprompt.py"):
        rates = KNOWPROMPT_LR_BY_K[k]
        return ["--lr1", str(rates["lr1"]), "--lr2", str(rates["lr2"])]

    return []


def span_split_args(model_path, span_split_eval, span_split_threshold):
    if not span_split_eval or model_path.endswith("baseline.py"):
        return []

    command_args = ["--span_split_eval"]
    if span_split_threshold is not None:
        command_args.extend(["--span_split_threshold", str(span_split_threshold)])
    return command_args


def should_skip_experiment(model_path, k, span_split_eval):
    if not model_path.endswith("baseline.py"):
        return False
    return span_split_eval or k != -1


def suite_runs(args):
    if args.suite == "replication":
        return [
            {"dataset": "semeval", "span_split_eval": False, "name": "semeval-main"},
            {"dataset": "causalnews", "span_split_eval": False, "name": "causalnews-main"},
            {"dataset": "causalnews", "span_split_eval": True, "name": "causalnews-span-split"},
        ]

    return [
        {
            "dataset": args.dataset,
            "span_split_eval": args.span_split_eval,
            "name": f"{args.dataset}{'-span-split' if args.span_split_eval else '-main'}",
        }
    ]


def main():
    args = parser.parse_args()
    if args.overwrite_results and os.path.exists(args.results_file):
        os.remove(args.results_file)

    # Capture the current environment and add the project root to PYTHONPATH
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    for run_config in suite_runs(args):
        dataset = run_config["dataset"]
        span_split_eval = run_config["span_split_eval"]
        print(f"\n=== Starting suite: {run_config['name']} ===")

        for seed in seeds:
            for k in k_values:
                for model_path in models:
                    if should_skip_experiment(model_path, k, span_split_eval):
                        print(f"\n>>> Skipping Experiment: {model_path} | dataset={dataset} | k={k} | seed={seed}")
                        continue

                    print(f"\n>>> Starting Experiment: {model_path} | dataset={dataset} | k={k} | seed={seed}")
                    command = [
                        sys.executable,
                        model_path,
                        "--dataset",
                        dataset,
                        "--k",
                        str(k),
                        "--seed",
                        str(seed),
                        "--results_file",
                        args.results_file,
                        "--experiment_name",
                        run_config["name"],
                        *learning_rate_args(model_path, k),
                        *span_split_args(model_path, span_split_eval, args.span_split_threshold),
                    ]
                    if args.no_wandb:
                        command.append("--no_wandb")
                    if args.use_wandb:
                        command.append("--use_wandb")

                    # Using subprocess to ensure a clean memory state for each run
                    subprocess.run(
                        command,
                        check=True,
                        env=env,
                    )


if __name__ == "__main__":
    main()
