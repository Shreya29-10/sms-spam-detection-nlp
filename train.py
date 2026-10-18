import re
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

# 1. Load data
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print("Data shape:", df.shape)

# 2. Clean text
def clean(t):
    return re.sub(r"[^a-z\s]", "", t.lower())

df["clean"] = df["text"].apply(clean)

# 3. Split + TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

tfidf = TfidfVectorizer(stop_words="english")
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# 4. Models (SVM ko calibrate karte hain taaki probability mil sake)
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(), cv=5),
}
files = {
    "Naive Bayes": "model_nb.pkl",
    "Logistic Regression": "model_lr.pkl",
    "Linear SVM": "model_svm.pkl",
}

metrics = {}
for name, m in models.items():
    m.fit(Xtr, y_train)
    pred = m.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    metrics[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    joblib.dump(m, files[name])

# 5. Results table
table = pd.DataFrame(metrics).T[["accuracy", "precision", "recall", "f1"]]
print("\nSpam class metrics:")
print(table)

nb_pred = models["Naive Bayes"].predict(Xte)
print("\nClassification report (Naive Bayes):")
print(classification_report(y_test, nb_pred, target_names=["Ham", "Spam"]))

# 6. Confusion matrix image (Naive Bayes)
ConfusionMatrixDisplay(confusion_matrix(y_test, nb_pred),
                       display_labels=["Ham", "Spam"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

# 7. Save everything
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(models["Naive Bayes"], "model.pkl")   # purane code ke liye
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("All models and metrics saved!")