
import re
import tkinter as tk
from tkinter import messagebox
import joblib

# Load the trained machine learning model
model = joblib.load("spam_email_model.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def check_email():
    subject = subject_entry.get().strip()
    body = body_text.get("1.0", tk.END).strip()

    if not subject and not body:
        messagebox.showwarning(
            "Missing Email",
            "Please enter an email subject or body."
        )
        return

    email_text = clean_text(subject + " " + body)
    result = model.predict([email_text])[0]

    if result == "spam":
        result_label.config(
            text="⚠ SPAM EMAIL",
            fg="#dc2626"
        )
    else:
        result_label.config(
            text="✓ NOT SPAM",
            fg="#15803d"
        )


def clear_fields():
    subject_entry.delete(0, tk.END)
    body_text.delete("1.0", tk.END)
    result_label.config(text="Prediction will appear here", fg="#374151")


# Main application window
root = tk.Tk()
root.title("AI Spam Email Detector")
root.geometry("700x620")
root.configure(bg="#f3f4f6")
root.resizable(False, False)

# Heading
title_label = tk.Label(
    root,
    text="AI SPAM EMAIL DETECTOR",
    font=("Arial", 22, "bold"),
    bg="#f3f4f6",
    fg="#1e3a8a"
)
title_label.pack(pady=(25, 5))

subtitle = tk.Label(
    root,
    text="Machine Learning Based Email Classification",
    font=("Arial", 11),
    bg="#f3f4f6",
    fg="#4b5563"
)
subtitle.pack(pady=(0, 20))

# Subject field
tk.Label(
    root,
    text="Email Subject",
    font=("Arial", 12, "bold"),
    bg="#f3f4f6",
    fg="#111827"
).pack(anchor="w", padx=45)

subject_entry = tk.Entry(
    root,
    font=("Arial", 12),
    width=65,
    relief="solid",
    bd=1
)
subject_entry.pack(padx=45, pady=(6, 18), ipady=7)

# Email body field
tk.Label(
    root,
    text="Email Body",
    font=("Arial", 12, "bold"),
    bg="#f3f4f6",
    fg="#111827"
).pack(anchor="w", padx=45)

body_text = tk.Text(
    root,
    height=8,
    width=65,
    font=("Arial", 11),
    wrap="word",
    relief="solid",
    bd=1
)
body_text.pack(padx=45, pady=(6, 18))

# Buttons
button_frame = tk.Frame(root, bg="#f3f4f6")
button_frame.pack(pady=5)

check_button = tk.Button(
    button_frame,
    text="CHECK EMAIL",
    command=check_email,
    font=("Arial", 11, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    width=18,
    pady=9,
    cursor="hand2"
)
check_button.pack(side="left", padx=10)

clear_button = tk.Button(
    button_frame,
    text="CLEAR",
    command=clear_fields,
    font=("Arial", 11, "bold"),
    bg="#e5e7eb",
    fg="#111827",
    width=12,
    pady=9,
    cursor="hand2"
)
clear_button.pack(side="left", padx=10)

# Prediction result
result_label = tk.Label(
    root,
    text="Prediction will appear here",
    font=("Arial", 17, "bold"),
    bg="#f3f4f6",
    fg="#374151"
)
result_label.pack(pady=20)

footer = tk.Label(
    root,
    text="Powered by TF-IDF and Multinomial Naive Bayes",
    font=("Arial", 9),
    bg="#f3f4f6",
    fg="#6b7280"
)
footer.pack(side="bottom", pady=12)

root.mainloop()
