# SMS and Email Spam Detection using NLP

Mini project for TE NLP (Sem-5), Dept. of AI & Data Science Engineering.
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spamshield-shreya.streamlit.app/)
## Problem Statement
Classify a given SMS/email text as Spam or Ham (not spam).

## Dataset
SMS Spam Collection Dataset (5,572 messages) - Kaggle / UCI.

## Approach
1. Text cleaning (lowercase, remove punctuation/numbers)
2. TF-IDF feature extraction
3. Models compared: Naive Bayes, Logistic Regression, Linear SVM
4. Streamlit web app for prediction

## Results

| Model | Accuracy | Precision | Recall | F1-score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 96.23% | 99.08% | 72.48% | 83.72% |
| Logistic Regression | 96.05% | 99.07% | 71.14% | 82.81% |
| Linear SVM | 98.39% | 97.12% | 90.60% | 93.75% |

Linear SVM gives the best F1-score. The confusion matrix shown below is for Naive Bayes.

![Confusion Matrix](confusion_matrix.png)

## How to Run
pip install -r requirements.txt
python train.py
streamlit run app.py

## Screenshots
