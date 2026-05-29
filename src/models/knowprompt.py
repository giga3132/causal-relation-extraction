from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaModel, get_scheduler, Trainer, TrainingArguments
from src.data.generate_k_shot import generate_k_shot_examples
from src.data.data import load_and_process
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
import evaluate
import wandb
import time

num_labels = 3
num_epochs = 5

def tokenize_function(examples):
    '''Tokenizes the input sentences with a prompt template.'''
    templated_sentences = [f"{s} The relation is {tokenizer.mask_token}." for s in examples["sentence"]]
    return tokenizer(templated_sentences, truncation=True, padding="max_length", max_length=128)

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
model = RobertaModel.from_pretrained("roberta-base")

tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
model.resize_token_embeddings(len(tokenizer))

# ── Verbalizer ──────────────────────────────────────────
answer_words = nn.Embedding(num_labels, 768)

relation_labels = ["Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)", "Other"]
label_descriptions = {
    "Cause-Effect(e1,e2)": "the first entity causes or produces the second entity",
    "Cause-Effect(e2,e1)": "the second entity causes or produces the first entity",
    "Other": "no causal or directional relation between the entities"
}

# Initialize answer_words embeddings
with torch.no_grad():
    for i, label in enumerate(relation_labels):
        description = label_descriptions[label]
        tokens = tokenizer(description, return_tensors="pt")
        outputs = model(**tokens)
        avg = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        answer_words.weight.data[i] = avg

#Verify initialization
v0 = answer_words.weight.data[0]  # Cause-Effect(e1,e2)
v1 = answer_words.weight.data[1]  # Cause-Effect(e2,e1)
v2 = answer_words.weight.data[2]  # Other

print(F.cosine_similarity(v0.unsqueeze(0), v1.unsqueeze(0)))
print(F.cosine_similarity(v0.unsqueeze(0), v2.unsqueeze(0)))

# ── Virtual type words ──────────────────────────────────────────
type_descriptions = {
    "<e1>":   ["start", "first", "entity"],
    "</e1>":  ["end", "first", "entity"],
    "<e2>":   ["start", "second", "entity"],
    "</e2>":  ["end", "second", "entity"],
}

with torch.no_grad():
    for marker, words in type_descriptions.items():
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        word_ids = []
        for word in words:
            word_ids.extend(tokenizer.encode(word, add_special_tokens=False))
        
        avg = model.embeddings.word_embeddings.weight.data[word_ids].mean(dim=0)
        model.embeddings.word_embeddings.weight.data[marker_id] = avg

# Verify initialization
e1_id = tokenizer.convert_tokens_to_ids("<e1>")
entity_id = tokenizer.encode("entity", add_special_tokens=False)[0]
e1_emb = model.embeddings.word_embeddings.weight.data[e1_id]
entity_emb = model.embeddings.word_embeddings.weight.data[entity_id]
print(f"Similarity <e1> to 'entity': {F.cosine_similarity(e1_emb.unsqueeze(0), entity_emb.unsqueeze(0)).item():.4f}")

# ── Tokenize datasets ───────────────────────────────────────────────────────
semeval = semeval.map(tokenize_function, batched=True,remove_columns="sentence")
semeval_k_train = semeval_k_train.map(tokenize_function, batched = True, remove_columns="sentence")

semeval.set_format("torch")
semeval_k_train.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── Dataloaders & optimizer ────────────────────────────────────────────────
k_train_dataloader = DataLoader(semeval_k_train, shuffle=True, batch_size=4, collate_fn=data_collator)

eval_dataloader = DataLoader(semeval["test"], batch_size=8, collate_fn=data_collator)

optimizer = AdamW(list(model.parameters()) + list(answer_words.parameters()), lr=5e-5)

num_training_steps = num_epochs * len(k_train_dataloader)
lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
answer_words.to(device)

# ── Training loop ──────────────────────────────────────────────────────────
with tqdm(range(num_training_steps), desc="Training", position=1, leave=True) as progress_bar:
    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────

        model.train()
        for batch in k_train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
             # batch["inputs_id"] is (batch_size, seq_length), take boolean array with positions of masks, and then take tensor with mask positions.
            mask_pos = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            # (batch_size, vocab_size)
            mask_hidden = outputs.last_hidden_state[torch.arange(batch["input_ids"].size(0)), mask_pos]

            logits = torch.matmul(mask_hidden, answer_words.weight.T)
            
            loss = nn.CrossEntropyLoss()(logits, batch["labels"])
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
                
                mask_pos = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                # (batch_size, vocab_size)
                mask_hidden = outputs.last_hidden_state[torch.arange(batch["input_ids"].size(0)), mask_pos]

                logits = torch.matmul(mask_hidden, answer_words.weight.T)
                
                loss = nn.CrossEntropyLoss()(logits, batch["labels"])

                eval_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
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
