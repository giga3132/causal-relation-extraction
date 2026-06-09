import math
from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaModel, get_scheduler, Trainer, TrainingArguments, set_seed
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval", help="Dataset to use: semeval or clean.")
parser.add_argument("--k", type=int, default=-1, help="k-shot size. Use -1 for full dataset.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=None, help="Fallback learning rate used for both parameter groups.")
parser.add_argument("--lr1", type=float, default=None, help="Learning rate for prompt/verbalizer answer words.")
parser.add_argument("--lr2", type=float, default=None, help="Learning rate for the base RoBERTa parameters.")
parser.add_argument("--max_length", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

set_seed(args.seed)

num_labels = 3
num_epochs = args.epochs
clean_dataset_names = {"clean", "local", "causal_clean"}
max_length = args.max_length if args.max_length is not None else (256 if args.dataset in clean_dataset_names else 128)
fallback_lr = args.lr if args.lr is not None else (5e-5 if args.k != -1 else 2e-5)
lr1 = args.lr1 if args.lr1 is not None else fallback_lr
lr2 = args.lr2 if args.lr2 is not None else fallback_lr

def tokenize_function(examples):
    '''Tokenizes the input sentences with a prompt template.'''
    templated_sentences = [f"{s} The relation is {tokenizer.mask_token}." for s in examples["sentence"]]
    return tokenizer(templated_sentences, truncation=True, padding="max_length", max_length=max_length)

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
dataset = load_and_process(args.dataset)

if args.k != -1:
    train_dataset = generate_k_shot_examples(dataset["train"], args.k)
else:
    train_dataset = dataset["train"]

print(f"Dataset: {args.dataset}")
print(f"Number of training examples: {len(train_dataset)}")
print(f"Max sequence length: {max_length}")
print(f"Learning rates: lr1={lr1}, lr2={lr2}")

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
dataset = dataset.map(tokenize_function, batched=True,remove_columns="sentence")
train_dataset = train_dataset.map(tokenize_function, batched = True, remove_columns="sentence")

dataset.set_format("torch")
train_dataset.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── Dataloaders & optimizer ────────────────────────────────────────────────
k_train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

eval_dataloader = DataLoader(dataset["test"], batch_size=8, collate_fn=data_collator)

optimizer = AdamW(
    [
        {"params": answer_words.parameters(), "lr": lr1},
        {"params": model.parameters(), "lr": lr2},
    ]
)

accumulation_steps = 4
scaler = torch.amp.GradScaler('cuda') # For FP16 speedup

updates_per_epoch = math.ceil(len(k_train_dataloader) / accumulation_steps)
num_training_steps = num_epochs * updates_per_epoch

lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)
print(f"Total training steps (with accumulation): {num_training_steps}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
answer_words.to(device)

run_name = f"knowprompt-{args.dataset}-k{args.k}-s{args.seed}" if args.k != -1 else f"knowprompt-{args.dataset}-full-s{args.seed}"
wandb_config = vars(args).copy()
wandb_config.update({"effective_lr1": lr1, "effective_lr2": lr2, "effective_max_length": max_length})
wandb.init(project="causal-re-final", name=run_name, config=wandb_config)

# ── Training loop ──────────────────────────────────────────────────────────
with tqdm(range(num_training_steps), desc="Training", position=1, leave=True) as progress_bar:
    global_step = 0
    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────
        model.train()
        answer_words.train()
        for i, batch in enumerate(k_train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda'): # Mixed Precision
                # batch["inputs_id"] is (batch_size, seq_length), take boolean array with positions of masks, and then take tensor with mask positions.
                mask_pos = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                
                # Extract mask hidden states
                mask_hidden = outputs.last_hidden_state[torch.arange(batch["input_ids"].size(0)), mask_pos]
                logits = torch.matmul(mask_hidden, answer_words.weight.T)
                loss = nn.CrossEntropyLoss()(logits, batch["labels"])
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1
                wandb.log({"train_loss": loss.item() * accumulation_steps, "iteration": global_step})
        

        # ── Evaluation ───────────────────────────────────

        model.eval()
        answer_words.eval()
        all_logits, all_labels = [], []
        eval_loss = 0.0
        num_eval_steps = len(eval_dataloader)
        num_eval_samples = len(dataset["test"])

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
        wandb.log({k: float(v) for k, v in eval_metrics.items() if k != "epoch"})
