
import re
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

print("Loading trained email spam detector...")
model = joblib.load("spam_email_model.pkl")
print("Model loaded successfully!")

while True:
    subject = input("\nEmail subject (or 'exit'): ").strip()

    if subject.lower() == "exit":
        print("Thank you for using the Email Spam Detector!")
        break

    body = input("Email body: ").strip()

    if not subject and not body:
        print("Please enter an email subject or body.")
        continue

    email_text = clean_text(subject + " " + body)
    result = model.predict([email_text])[0]

    if result == "spam":
        print("Prediction: SPAM")
    else:
        print("Prediction: NOT SPAM")
