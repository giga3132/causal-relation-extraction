import subprocess
import os
import sys

# -1 represents full shot (the whole dataset)
k_values = [8, 16, 32, 64, 128, 256, -1]
seeds = [42, 43, 44, 45, 46]
models = [
    "src/models/baseline.py",
    "src/models/roberta.py",
    "src/models/knowprompt.py"
]

# Capture the current environment and add the project root to PYTHONPATH
env = os.environ.copy()
project_root = os.path.dirname(os.path.abspath(__file__))
env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")

for seed in seeds:
    for k in k_values:
        for model_path in models:
            print(f"\n>>> Starting Experiment: {model_path} | k={k} | seed={seed}")
            # Using subprocess to ensure a clean memory state for each run
            subprocess.run([sys.executable, model_path, "--k", str(k), "--seed", str(seed)], check=True, env=env)
