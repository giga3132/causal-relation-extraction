from transformers import DataCollatorWithPadding, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, set_seed
from src.data.generate_k_shot import generate_k_shot_examples
from src.data.data import load_and_process
from src.data.span_splits import split_by_entity_span_length
import numpy as np
import evaluate
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval", help="Dataset to use: semeval or clean.")
parser.add_argument("--k", type=int, default=-1, help="k-shot size. Use -1 for full dataset.")
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--span_split_eval", action="store_true", help="Evaluate clean test examples separately by entity span length.")
parser.add_argument("--span_split_threshold", type=float, default=None, help="Short/long threshold over max entity span length. Defaults to the test median.")
args = parser.parse_args()

set_seed(args.seed)

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
dataset = load_and_process(args.dataset)

if args.k != -1:
    train_dataset = generate_k_shot_examples(dataset["train"], args.k)
else:
    train_dataset = dataset["train"]

print(f"Dataset: {args.dataset}")
print(f"Number of training examples: {len(train_dataset)}")

span_eval_datasets = {}
span_threshold = None
span_counts = None
if args.span_split_eval:
    span_eval_datasets, span_threshold, span_counts = split_by_entity_span_length(
        dataset["test"],
        threshold=args.span_split_threshold,
    )
    print(
        f"Span split threshold: max entity span <= {span_threshold} is short; "
        f"short={span_counts['short']}, long={span_counts['long']}"
    )

# Load metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

tokenizer.add_special_tokens({"additional_special_tokens": ["<e1>", "</e1>", "<e2>", "</e2>"]})
model.resize_token_embeddings(len(tokenizer))

dataset = dataset.map(tokenize_function, batched=True,)
train_dataset = train_dataset.map(tokenize_function, batched=True,)
span_eval_datasets = {
    name: split.map(tokenize_function, batched=True)
    for name, split in span_eval_datasets.items()
}
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Initialize wandb for experiment tracking
run_name = f"roberta-{args.dataset}-k{args.k}-s{args.seed}" if args.k != -1 else f"roberta-{args.dataset}-full-s{args.seed}"
wandb.init(project="causal-re-final-split", name=run_name, config=args)


# Training

training_args = TrainingArguments(f"outputs/roberta-{args.dataset}", 
                                  eval_strategy="epoch",
                                  logging_steps=20,
                                  learning_rate=args.lr,
                                  num_train_epochs=args.epochs,
                                  per_device_train_batch_size=4,
                                  gradient_accumulation_steps=4,
                                  fp16=True,
                                  seed=args.seed,
                                #   report_to="wandb"
                                  report_to="wandb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()

if args.span_split_eval:
    wandb.log({
        "span_split_threshold": span_threshold,
        "span_split_short_n": span_counts["short"],
        "span_split_long_n": span_counts["long"],
    })
    for split_name, eval_dataset in span_eval_datasets.items():
        metrics = trainer.evaluate(
            eval_dataset=eval_dataset,
            metric_key_prefix=f"eval_{split_name}_span",
        )
        print(f"{split_name.title()} span evaluation: {metrics}")
