import joblib
import os
import re
import email
from email import policy
from email.parser import BytesParser

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATASET_PATH = "dataset"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_email_text(message):
    parts = []

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_type() == "text/plain":
                try:
                    content = part.get_content()
                    if isinstance(content, str):
                        parts.append(content)
                except Exception:
                    continue
    else:
        try:
            content = message.get_content()
            if isinstance(content, str):
                parts.append(content)
        except Exception:
            pass

    return "\n".join(parts)

def load_emails(folder, label):
    emails = []
    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        print("Folder not found:", folder_path)
        return emails

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if not os.path.isfile(file_path):
            continue

        try:
            with open(file_path, "rb") as file:
                message = BytesParser(
                    policy=policy.default
                ).parse(file)

            subject = str(message.get("subject", ""))
            body = get_email_text(message)
            text = clean_text(subject + " " + body)

            if text:
                emails.append({
                    "label": label,
                    "message": text
                })

        except Exception as error:
            print("Skipped unreadable file:", filename)

    return emails

print("Loading real email dataset...")

all_emails = (
    load_emails("easy_ham", "ham")
    + load_emails("spam_2", "spam")
)

data = pd.DataFrame(all_emails)

if data.empty:
    raise ValueError(
        "No emails loaded. Check your dataset folders."
    )

data = data.drop_duplicates(subset=["message"])
data = data[data["message"].str.len() > 0]

print("\nTotal emails:", len(data))
print("Legitimate emails:", (data["label"] == "ham").sum())
print("Spam emails:", (data["label"] == "spam").sum())

X = data["message"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000
    )),
    ("classifier", MultinomialNB(alpha=0.5))
])

print("\nTraining email classifier...")
model.fit(X_train, y_train)
joblib.dump(model, "spam_email_model.pkl")
print("Trained model saved as spam_email_model.pkl")
predictions = model.predict(X_test)

print("\nModel Accuracy:",
      round(accuracy_score(y_test, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    target_names=["Not Spam", "Spam"],
    zero_division=0
))

print("\nConfusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    y_test, predictions, labels=["ham", "spam"]
))

print("\n--- Spam Email Detection ---")
print("Enter an email subject and body, or type 'exit'.")

while True:
    subject = input("\nEmail subject: ").strip()

    if subject.lower() == "exit":
        print("Thank you for testing the Email Spam Detector!")
        break

    body = input("Email body: ").strip()

    if not subject and not body:
        print("Please enter a subject or email body.")
        continue

    email_text = clean_text(subject + " " + body)
    result = model.predict([email_text])[0]

    if result == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")
