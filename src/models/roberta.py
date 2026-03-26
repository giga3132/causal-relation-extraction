from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from src.data.data import load_and_process
import numpy as np
import evaluate


def tokenize_function(examples):
    '''Tokenizes the input sentences.'''
    return tokenizer(examples["sentence"], padding="max_length", truncation=True)

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"]
    }


# Load and preprocess the dataset
semeval = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8")
print(semeval["train"][0])

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
model = RobertaForSequenceClassification.from_pretrained("distilroberta-base", num_labels=3)

tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
model.resize_token_embeddings(len(tokenizer))

semeval = semeval.map(tokenize_function, batched=True,)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Training

training_args = TrainingArguments("outputs/roberta", 
                                  eval_strategy="steps",
                                  eval_steps=0.5,
                                  logging_dir="outputs/roberta/logs",
                                  logging_steps=10,
                                  num_train_epochs=0.5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=semeval["train"],
    eval_dataset=semeval["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()