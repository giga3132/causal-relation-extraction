from collections import Counter, defaultdict
import numpy as np

def generate_k_shot_examples(dataset, k):
    """Generates k-shot examples from the dataset."""

    indexes_per_class = defaultdict(list)
    for idx, label in enumerate(dataset["labels"]):
        indexes_per_class[label].append(idx)

    selected_indexes = []
    for label, indexes in indexes_per_class.items():
        selected_indexes.extend(np.random.choice(indexes, size=k, replace=False))

    k_shot_examples = dataset.select(selected_indexes)

    return k_shot_examples