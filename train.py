import re
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# 1. Load data
df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})
print("Data shape:", df.shape)
print(df["label"].value_counts())

# 2. Clean text
def clean(t):
    t = t.lower()
    t = re.sub(r"[^a-z\s]", "", t)
    return t

df["clean"] = df["text"].apply(clean)

# 3. Split + TF-IDF
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

tfidf = TfidfVectorizer(stop_words="english")
Xtr = tfidf.fit_transform(X_train)
Xte = tfidf.transform(X_test)

# 4. Train 3 models and compare
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

for name, m in models.items():
    m.fit(Xtr, y_train)
    acc = accuracy_score(y_test, m.predict(Xte))
    print(f"{name}: accuracy = {acc:.4f}")

# 5. Detailed report for the main model (Naive Bayes)
model = models["Naive Bayes"]
pred = model.predict(Xte)
print("\nClassification report (Naive Bayes):")
print(classification_report(y_test, pred, target_names=["Ham", "Spam"]))

# 6. Confusion matrix image (poster/report ke liye)
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"]).plot(cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

# 7. Save model
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
print("Model saved!")