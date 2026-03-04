from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification

ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")

#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

for i in range(50):
    print(ds["train"][i])