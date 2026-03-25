from datasets import load_dataset

def load_and_process(dataset_name):
    dataset = load_dataset(dataset_name)


    def _collapse_relations(batch):
        def map_rel(r):
            return 0 if r == 0 else (1 if r ==  1 else 2)
        return {"relation": [map_rel(r) for r in batch["relation"]]}

    dataset = dataset.map(_collapse_relations, batched=True)

    return dataset