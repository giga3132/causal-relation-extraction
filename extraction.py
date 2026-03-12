from datasets import load_dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
import pandas as pd


ds = load_dataset("SemEvalWorkshop/sem_eval_2010_task_8")
nb = MultinomialNB(alpha=1.0)

vectorizer = CountVectorizer(
    max_features=10000, 
    ngram_range=(1, 2),   
    stop_words="english"  
)

X_train_vec = vectorizer.fit_transform(ds["train"]["sentence"])
X_test_vec = vectorizer.transform(ds["test"]["sentence"])

y_train = ds["train"]["relation"]
y_test = ds["test"]["relation"]

nb.fit(X_train_vec, y_train)

y_pred = nb.predict(X_test_vec)

#print(classification_report(y_test, y_pred, 
#                             target_names=list(ds["train"]["relation"].keys())))

macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.4f}")



#tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)

#for i in range(2):
#    print(ds["train"]["relation"][i])

