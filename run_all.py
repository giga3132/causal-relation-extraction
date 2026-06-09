import subprocess
import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval", help="Dataset to use: semeval or clean.")

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


def main():
    args = parser.parse_args()

    # Capture the current environment and add the project root to PYTHONPATH
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

    for seed in seeds:
        for k in k_values:
            for model_path in models:
                print(f"\n>>> Starting Experiment: {model_path} | dataset={args.dataset} | k={k} | seed={seed}")
                command = [
                    sys.executable,
                    model_path,
                    "--dataset",
                    args.dataset,
                    "--k",
                    str(k),
                    "--seed",
                    str(seed),
                    *learning_rate_args(model_path, k),
                ]
                # Using subprocess to ensure a clean memory state for each run
                subprocess.run(
                    command,
                    check=True,
                    env=env,
                )


if __name__ == "__main__":
    main()
