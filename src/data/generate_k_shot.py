from collections import Counter
import numpy as np

def generate_k_shot_examples(dataset, k):
    """Generates k-shot examples from the dataset."""

    np.random.seed(42)  # For reproducibility
    dataset = dataset.shuffle(seed=42)

    full_counts = Counter(dataset["labels"])
    print("Full count:" + str(full_counts))
    counts = {label: 0 for label in full_counts}
    print("Zero count:" + str(counts))

    k_shot_examples = []
    for label in counts:
        while counts[label] < k:
            k_shot_examples.append(dataset.filter(lambda x: x["labels"] == label)[counts[label]])
            counts[label] += 1
    print("K-shot examples:" + str(k_shot_examples))
    return k_shot_examples