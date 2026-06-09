import csv
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DATA_DIR = PROJECT_ROOT / "src" / "data" / "datasets"

LABEL_TO_ID = {
    "Cause-Effect(e1,e2)": 0,
    "Cause-Effect(e2,e1)": 1,
    "Other": 2,
}


def _normalise_label(label):
    return label.strip().rstrip(".")


def _has_valid_entity_tags(sentence):
    tags = ("<e1>", "</e1>", "<e2>", "</e2>")
    if any(sentence.count(tag) != 1 for tag in tags):
        return False

    e1_start = sentence.index("<e1>")
    e1_end = sentence.index("</e1>")
    e2_start = sentence.index("<e2>")
    e2_end = sentence.index("</e2>")

    return e1_start < e1_end and e2_start < e2_end


def _load_clean_split(path):
    rows = {"sentence": [], "labels": []}
    skipped_rows = 0

    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file, delimiter="\t", quotechar='"')
        for line_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) != 3:
                raise ValueError(f"Expected 3 tab-separated columns in {path}:{line_number}, got {len(row)}")

            _, sentence, label = row
            sentence = sentence.replace("\ufeff", "").strip()
            if not _has_valid_entity_tags(sentence):
                skipped_rows += 1
                print(f"Warning: skipping malformed entity tags in {path}:{line_number}")
                continue

            label = _normalise_label(label)
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown relation label {label!r} in {path}:{line_number}")

            rows["sentence"].append(sentence)
            rows["labels"].append(LABEL_TO_ID[label])

    if skipped_rows:
        print(f"Warning: skipped {skipped_rows} malformed rows from {path}")

    return Dataset.from_dict(rows)


def _load_clean_dataset():
    return DatasetDict(
        {
            "train": _load_clean_split(LOCAL_DATA_DIR / "train_clean.txt"),
            "test": _load_clean_split(LOCAL_DATA_DIR / "dev_clean.txt"),
        }
    )


def _load_semeval_dataset(dataset_name):
    dataset = load_dataset(dataset_name)

    def _collapse_relations(batch):
        def map_rel(r):
            return 0 if r == 0 else (1 if r ==  1 else 2)
        return {"relation": [map_rel(r) for r in batch["relation"]]}
    

    dataset = dataset.map(_collapse_relations, batched=True)
    dataset = dataset.rename_column("relation", "labels")

    return dataset


def load_and_process(dataset_name):
    if dataset_name in {"clean", "causalnews", "causal_clean"}:
        return _load_clean_dataset()

    if dataset_name in {"semeval", "SemEvalWorkshop/sem_eval_2010_task_8"}:
        return _load_semeval_dataset("SemEvalWorkshop/sem_eval_2010_task_8")

    return _load_semeval_dataset(dataset_name)
