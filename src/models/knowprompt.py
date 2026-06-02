import math
import re
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
parser.add_argument("--k", type=int, default=-1, help="k-shot size. Use -1 for full dataset.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────

set_seed(args.seed)

num_labels = 3
num_epochs = args.epochs

# ── Helper functions ────────────────────────────────────────────────

def tokenize_function(examples):
    templated_sentences = []
    for s in examples["sentence"]:
        # Extract entity text from inline markers
        e1_match = re.search(r'<e1>(.*?)</e1>', s)
        e2_match = re.search(r'<e2>(.*?)</e2>', s)
        e1_text = e1_match.group(1).strip() if e1_match else ""
        e2_text = e2_match.group(1).strip() if e2_match else ""

        # Strip all markers to get clean sentence
        clean = re.sub(r'</?e[12]>', '', s).strip()

        # Rebuild in new format
        templated = f"{clean} <V1> {e1_text} <V1> {tokenizer.mask_token} <V2> {e2_text} <V2>"
        templated_sentences.append(templated)

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

def transE_loss(V1_h, rel_h, V2_h, gamma=1.0):
    """Computes the TransE structural constraint: ||s + r - o||"""
    pos_score = torch.norm(V1_h + rel_h - V2_h, p=2, dim=-1)
    # Negative: shuffle V2 within batch to create 'corrupted' triples
    e2_neg = V2_h[torch.randperm(V2_h.size(0))]
    neg_score = torch.norm(V1_h + rel_h - e2_neg, p=2, dim=-1)
    return -F.logsigmoid(gamma - pos_score).mean() - F.logsigmoid(neg_score - gamma).mean()

def get_model_outputs(batch, model, virtual_types, type_ids):
    inputs_embeds = model.embeddings.word_embeddings(batch["input_ids"]).clone()
    for i, tid in enumerate(type_ids):
        mask = (batch["input_ids"] == tid).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = torch.where(mask, virtual_types[i].expand_as(inputs_embeds), inputs_embeds)
    return model(inputs_embeds=inputs_embeds, attention_mask=batch["attention_mask"])

def evaluate_stage(epoch, stage, model, answer_words, virtual_type_embeddings, type_token_ids, eval_dataloader, device):
    """Standard evaluation loop."""
    model.eval(); answer_words.eval()
    all_logits, all_labels = [], []
    eval_loss, eval_start = 0.0, time.time()
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc=f"Evaluating Stage {stage}", position=1, leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = get_model_outputs(batch, model, virtual_type_embeddings, type_token_ids)
            mask_pos = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
            mask_h = outputs.last_hidden_state[mask_pos[:, 0], mask_pos[:, 1]]
            logits = torch.matmul(mask_h, answer_words.weight.T)
            eval_loss += nn.CrossEntropyLoss()(logits, batch["labels"]).item()
            all_logits.append(logits.cpu().numpy()); all_labels.append(batch["labels"].cpu().numpy())

    metrics = compute_metrics((np.concatenate(all_logits), np.concatenate(all_labels)))
    res = {"eval_loss": eval_loss/len(eval_dataloader), "eval_accuracy": metrics['accuracy'], "eval_f1": metrics['f1'], "stage": stage, "epoch": epoch+1}
    print(res); wandb.log(res)

def train_stage(stage_num, epochs, optimizer_params, use_struct_loss, lr, model, answer_words, virtual_type_embeddings, type_token_ids, train_dataloader, eval_dataloader, device):
    """Generic training stage to support two-phase optimization."""
    optimizer = AdamW(optimizer_params, lr=lr)
    accumulation_steps = 4
    num_training_steps = epochs * math.ceil(len(train_dataloader) / accumulation_steps)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n>>> Starting Stage {stage_num}: {epochs} epochs")
    
    progress_bar = tqdm(total=epochs * len(train_dataloader), desc=f"Stage {stage_num}", position=0, leave=True)
    for epoch in range(epochs):
        progress_bar.set_description(f"Stage {stage_num} | Epoch {epoch+1}/{epochs}")
        model.train() 
        answer_words.train()
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast('cuda'):
                outputs = get_model_outputs(batch, model, virtual_type_embeddings, type_token_ids)
                mask_pos = (batch["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=False)
                mask_h = outputs.last_hidden_state[mask_pos[:, 0], mask_pos[:, 1]]
                logits = torch.matmul(mask_h, answer_words.weight.T)
                
                loss = nn.CrossEntropyLoss()(logits, batch["labels"])
                if use_struct_loss:
                    V1_pos = (batch["input_ids"] == V1_id).nonzero(as_tuple=False)
                    V2_pos = (batch["input_ids"] == V2_id).nonzero(as_tuple=False)
                    V1_h = outputs.last_hidden_state[V1_pos[::2, 0], V1_pos[::2, 1]]
                    V2_h = outputs.last_hidden_state[V2_pos[::2, 0], V2_pos[::2, 1]]
                    loss += 0.001 * transE_loss(V1_h, mask_h, V2_h) #Gamma of 0.001 from paper.
                
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_dataloader):
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad()
                wandb.log({f"stage{stage_num}_loss": loss.item() * accumulation_steps})
            progress_bar.update(1)
        
        evaluate_stage(epoch, stage_num, model, answer_words, virtual_type_embeddings, type_token_ids, eval_dataloader, device)
    progress_bar.close()

# ── Data and model loading ──────────────────────────────────────────────────

# Load and preprocess the dataset
semeval = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8")

if args.k != -1:
    semeval_k_train = generate_k_shot_examples(semeval["train"], args.k)
else:
    semeval_k_train = semeval["train"]

print(f"Number of training examples: {len(semeval_k_train)}")

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")

tokenizer.add_special_tokens({"additional_special_tokens": ["<V1>", "<V2>"]})
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
    "<V1>":   ["Cause", "Effect"],
    "<V2>":  ["Cause", "Effect"],
}

with torch.no_grad():
    for marker, words in type_descriptions.items():
        marker_id = tokenizer.convert_tokens_to_ids(marker)
        word_ids = []
        for word in words:
            word_ids.extend(tokenizer.encode(word, add_special_tokens=False))
        
        avg = model.embeddings.word_embeddings.weight.data[word_ids].mean(dim=0)
        noise = torch.randn_like(avg) * 0.01
        model.embeddings.word_embeddings.weight.data[marker_id] = avg + noise

# Isolate virtual type words as standalone parameters
type_tokens = ["<V1>", "<V2>"]
type_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in type_tokens]
virtual_type_embeddings = nn.Parameter(model.embeddings.word_embeddings.weight.data[type_token_ids].clone())

# Verify initialization
V1_id = tokenizer.convert_tokens_to_ids("<V1>")
V2_id = tokenizer.convert_tokens_to_ids("<V2>")
entity_id = tokenizer.encode("entity", add_special_tokens=False)[0]
V1_emb = model.embeddings.word_embeddings.weight.data[V1_id]
entity_emb = model.embeddings.word_embeddings.weight.data[entity_id]
print(f"Similarity V1 to 'entity': {F.cosine_similarity(V1_emb.unsqueeze(0), entity_emb.unsqueeze(0)).item():.4f}")

# ── Tokenize datasets ───────────────────────────────────────────────────────
semeval = semeval.map(tokenize_function, batched=True, remove_columns="sentence")
semeval_k_train = semeval_k_train.map(tokenize_function, batched = True, remove_columns="sentence")

semeval.set_format("torch")
semeval_k_train.set_format("torch")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ── Dataloaders & optimizer ────────────────────────────────────────────────
k_train_dataloader = DataLoader(semeval_k_train, shuffle=True, batch_size=4, collate_fn=data_collator)

eval_dataloader = DataLoader(semeval["test"], batch_size=8, collate_fn=data_collator)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
answer_words.to(device)
virtual_type_embeddings = nn.Parameter(virtual_type_embeddings.data.to(device))

run_name = f"knowprompt-k{args.k}-s{args.seed}" if args.k != -1 else f"knowprompt-full-s{args.seed}"
wandb.init(project="causal-re", name=run_name, config=args)

# ── Execution ───────────────────────────────────────────────────────────────

# Stage 1: Anchor the Virtual words using Structural Loss
for param in model.parameters(): param.requires_grad_(False)
train_stage(
    stage_num=1, 
    epochs=2, 
    optimizer_params=[{"params": answer_words.parameters()}, {"params": [virtual_type_embeddings]}], 
    use_struct_loss=True, 
    lr=1e-4,
    model=model,
    answer_words=answer_words,
    virtual_type_embeddings=virtual_type_embeddings,
    type_token_ids=type_token_ids,
    train_dataloader=k_train_dataloader,
    eval_dataloader=eval_dataloader,
    device=device
)
# Stage 2: Fine-tune the Full model
for param in model.parameters(): param.requires_grad_(True)
train_stage(
    stage_num=2, 
    epochs=3, 
    optimizer_params=[
        {"params": model.parameters()},
        {"params": answer_words.parameters()},
        {"params": [virtual_type_embeddings]}
    ], 
    use_struct_loss=False, 
    lr=2e-5,
    model=model,
    answer_words=answer_words,
    virtual_type_embeddings=virtual_type_embeddings,
    type_token_ids=type_token_ids,
    train_dataloader=k_train_dataloader,
    eval_dataloader=eval_dataloader,
    device=device
)
