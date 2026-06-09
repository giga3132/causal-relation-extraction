from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import compute_sample_weight
from src.data.data import load_and_process
from src.data.generate_k_shot import generate_k_shot_examples
import argparse
import numpy as np
import wandb

def parse_sentence(sentence):
    """Extract tokens and entity spans from SemEval formatted sentences."""
    
    tokens = []
    e1s = e1e = e2s = e2e = None
    
    temp = sentence
    temp = temp.replace('<e1>', ' <e1> ').replace('</e1>', ' </e1> ')
    temp = temp.replace('<e2>', ' <e2> ').replace('</e2>', ' </e2> ')
    
    raw_tokens = temp.split()
    
    for tok in raw_tokens:
        if tok == '<e1>':
            e1s = len(tokens)
        elif tok == '</e1>':
            e1e = len(tokens) - 1
        elif tok == '<e2>':
            e2s = len(tokens)
        elif tok == '</e2>':
            e2e = len(tokens) - 1
        else:
            tokens.append(tok.lower())
    
    return tokens, (e1s, e1e), (e2s, e2e)


def extract_features(tokens):
    tokens, (e1s, e1e), (e2s, e2e) = parse_sentence(tokens)

    features = {}
    for w in tokens[max(0, e1s - 2):e1s]:
        features[f"e1_l:{w}"] = 1
    for w in tokens[e1s:e1e + 1]:
        features[f"e1:{w}"] = 1
    for w in tokens[e1e + 1:min(len(tokens), e1e + 3)]:
        features[f"e1_r:{w}"] = 1
    for w in tokens[max(0, e2s - 2):e2s]:
        features[f"e2_l:{w}"] = 1
    for w in tokens[e2s:e2e + 1]:
        features[f"e2:{w}"] = 1
    for w in tokens[e2e + 1:min(len(tokens), e2e + 3)]:
        features[f"e2_r:{w}"] = 1
    
    return features

def collapse_label(l):
    if l == 18:
        return 9
    return l // 2


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval", help="Dataset to use: semeval or clean.")
parser.add_argument("--k", type=int, default=-1, help="k-shot size. Use -1 for full dataset.")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
np.random.seed(args.seed)

dataset = load_and_process(args.dataset)

if args.k != -1:
    train_set = generate_k_shot_examples(dataset["train"], args.k)
    print(f"Running baseline on {args.dataset} with k={args.k}")
else:
    train_set = dataset["train"]
    print(f"Running baseline on {args.dataset} with full dataset")

nb = MultinomialNB(alpha=1.0)
dv = DictVectorizer(sparse=True)

train_dicts = [extract_features(s) for s in train_set["sentence"]]
test_dicts  = [extract_features(s) for s in dataset["test"]["sentence"]]

X_train_vec = dv.fit_transform(train_dicts)
print(X_train_vec.shape)
X_test_vec  = dv.transform(test_dicts)

y_train = train_set["labels"]
y_test  = dataset["test"]["labels"]

# sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
nb.fit(X_train_vec, y_train)

y_pred = nb.predict(X_test_vec)

print(classification_report(y_test, y_pred))

run_name = f"baseline-{args.dataset}-k{args.k}-s{args.seed}" if args.k != -1 else f"baseline-{args.dataset}-full-s{args.seed}"
wandb.init(project="causal-re-final", name=run_name, config=args)

labels = [l for l in nb.classes_ if l != 2]
macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels) #Macro F1 without "Other" class

metrics = {
    "eval_accuracy": accuracy_score(y_test, y_pred),
    "eval_f1": macro_f1
}
wandb.log(metrics)


# Evaluate F1 on collapsed labels for full classification task (9 labels)
# y_test_collapsed = [collapse_label(l) for l in y_test]
# y_pred_collapsed = [collapse_label(l) for l in y_pred]

# labels_collapsed = list(range(9))
# macro_f1_collapsed = f1_score(y_test_collapsed, y_pred_collapsed, average="macro", labels=labels_collapsed)
# print(f"Macro F1 collapsed (excl. Other): {macro_f1_collapsed:.4f}")
