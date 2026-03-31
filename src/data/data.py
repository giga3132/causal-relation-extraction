from datasets import load_dataset

def load_and_process(dataset_name, variant):
    dataset = load_dataset(dataset_name)

    def _collapse_relations(batch):
        def map_rel(r):
            return 0 if r == 0 else (1 if r ==  1 else 2)
        return {"relation": [map_rel(r) for r in batch["relation"]]}
    
    def _to_lower(batch):
       return {"sentence": [s.lower() for s in batch["sentence"]]}

    if variant == "nb_collapsed":
        dataset = dataset.map(_to_lower, batched=True)
        dataset = dataset.map(_collapse_relations, batched=True)
    elif variant == "nb_full":
        dataset = dataset.map(_to_lower, batched=True)
    elif variant == "roberta_collapsed":
        dataset = dataset.map(_collapse_relations, batched=True)
    dataset = dataset.rename_column("relation", "labels")

    return dataset