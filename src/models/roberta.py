from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from src.data.data import load_and_process

semeval = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8")



tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)


def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

semeval = semeval.map(tokenize_function, batched=True,)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=semeval["train"],
    eval_dataset=semeval["test"],
    data_collator=data_collator,
)

trainer.train()
