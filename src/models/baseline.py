from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
from src.data.data import load_and_process

semeval_full = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8", "nb_full")
semeval_collapsed = load_and_process("SemEvalWorkshop/sem_eval_2010_task_8", "nb_collapsed")

nb = MultinomialNB(alpha=1)

vectorizer = CountVectorizer(
    max_features=10000, 
    ngram_range=(1, 1),
)

X_train_vec = vectorizer.fit_transform(semeval_full["train"]["sentence"])
X_test_vec = vectorizer.transform(semeval_full["test"]["sentence"])

y_train = semeval_full["train"]["labels"]
y_test = semeval_full["test"]["labels"]

nb.fit(X_train_vec, y_train)

y_pred = nb.predict(X_test_vec)


print(classification_report(y_test, y_pred))

macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1: {macro_f1:.4f}")