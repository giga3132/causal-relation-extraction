from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import compute_sample_weight
from src.data.data import load_and_process

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


semeval = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8")

nb = MultinomialNB(alpha=1.0)
dv = DictVectorizer(sparse=True)

train_dicts = [extract_features(s) for s in semeval["train"]["sentence"]]
test_dicts  = [extract_features(s) for s in semeval["test"]["sentence"]]

X_train_vec = dv.fit_transform(train_dicts)
print(X_train_vec.shape)
X_test_vec  = dv.transform(test_dicts)

y_train = semeval["train"]["labels"]
y_test  = semeval["test"]["labels"]

# sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
nb.fit(X_train_vec, y_train)

y_pred = nb.predict(X_test_vec)

print(classification_report(y_test, y_pred))

labels = [l for l in nb.classes_ if l != 2]
macro_f1 = f1_score(y_test, y_pred, average="macro", labels=labels)
print(f"Macro F1 (excl. Other): {macro_f1:.4f}")

# Evaluate F1 on collapsed labels for full classification task (9 labels)
# y_test_collapsed = [collapse_label(l) for l in y_test]
# y_pred_collapsed = [collapse_label(l) for l in y_pred]

# labels_collapsed = list(range(9))
# macro_f1_collapsed = f1_score(y_test_collapsed, y_pred_collapsed, average="macro", labels=labels_collapsed)
# print(f"Macro F1 collapsed (excl. Other): {macro_f1_collapsed:.4f}")