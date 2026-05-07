from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from src.data.generate_k_shot import generate_k_shot_examples
from src.data.data import load_and_process
import numpy as np
import evaluate
import wandb


def tokenize_function(examples):
    '''Tokenizes the input sentences.'''
    return tokenizer(examples["sentence"], truncation=True)

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
semeval_k_train = generate_k_shot_examples(semeval["train"], 256)
print(f"Number of training examples: {len(semeval_k_train)}")

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
model.resize_token_embeddings(len(tokenizer))

semeval = semeval.map(tokenize_function, batched=True,)
semeval_k_train = semeval_k_train.map(tokenize_function, batched=True,)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Initialize wandb for experiment tracking
# wandb.init(project="transformer-fine-tuning", name="roberta-test")


# Training

training_args = TrainingArguments("outputs/roberta", 
                                  eval_strategy="epoch",
                                  logging_steps=20,
                                  num_train_epochs=5,
                                  per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4,
                                  fp16=True,
                                #   report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=semeval_k_train,
    eval_dataset=semeval["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()