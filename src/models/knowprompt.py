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
semeval_k_train = generate_k_shot_examples(semeval["train"], 16)

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

k_train_dataloader = DataLoader(semeval_k_train, shuffle=True, batch_size=8, collate_fn=data_collator)

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

device = torch.device("cuda") 
model.to(device)


model.train()
for epoch in range(num_epochs):
    progress_bar = tqdm(range(len(k_train_dataloader)), leave=False)
    for batch in k_train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)



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

