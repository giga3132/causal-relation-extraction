from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, get_scheduler, Trainer, TrainingArguments
from src.data.generate_k_shot import generate_k_shot_examples
from src.data.data import load_and_process
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import evaluate
import wandb
import time


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

# Tokenize datasets
semeval = semeval.map(tokenize_function, batched=True,remove_columns="sentence")
semeval_k_train = semeval_k_train.map(tokenize_function, batched = True, remove_columns="sentence")

semeval.set_format("torch")
semeval_k_train.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

k_train_dataloader = DataLoader(semeval_k_train, shuffle=True, batch_size=16, collate_fn=data_collator)

eval_dataloader = DataLoader(semeval["test"], batch_size=8, collate_fn=data_collator)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 5
num_training_steps = num_epochs * len(k_train_dataloader)
lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


with tqdm(range(num_training_steps), desc="Training", position=1, leave=True) as progress_bar:
    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────

        model.train()
        for batch in k_train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # ── Evaluation ───────────────────────────────────

        model.eval()
        all_logits, all_labels = [], []
        eval_loss = 0.0
        num_eval_steps = len(eval_dataloader)
        num_eval_samples = len(semeval["test"])

        eval_bar = tqdm(eval_dataloader, 
                        desc=f"Evaluating epoch {epoch + 1}", 
                        position=0, 
                        leave=False)

        eval_start = time.time()

        with torch.no_grad():
            for batch in eval_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                eval_loss += outputs.loss.item()
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        eval_runtime = time.time() - eval_start

        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        metrics = compute_metrics((all_logits, all_labels))

        eval_metrics = {
            "eval_loss":                 f"{eval_loss / num_eval_steps:.4f}",
            "eval_accuracy":             f"{metrics['accuracy']:.4f}",
            "eval_f1":                   f"{metrics['f1']:.4f}",
            "eval_runtime":              f"{eval_runtime:.3f}",
            "eval_samples_per_second":   f"{num_eval_samples / eval_runtime:.2f}",
            "eval_steps_per_second":     f"{num_eval_steps / eval_runtime:.2f}",
            "epoch":                     f"{epoch + 1}",
        }

        tqdm.write(str(eval_metrics))



# Initialize wandb for experiment tracking
# wandb.init(project="transformer-fine-tuning", name="knowprompt-proto")


# Training


# training_args = TrainingArguments("outputs/roberta", 
#                                   eval_strategy="epoch",
#                                   logging_steps=20,
#                                   num_train_epochs=5,
#                                   per_device_train_batch_size=4,
#                                   gradient_accumulation_steps=4,
#                                   fp16=True,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=semeval_k_train,
#     eval_dataset=semeval["test"],
#     data_collator=data_collator,
#     processing_class=tokenizer,
#     compute_metrics=compute_metrics,
# )

