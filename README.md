# Spam Email Detection Using Machine Learning

## Project Overview
This project is a machine learning application that classifies emails as **Spam** or **Not Spam** based on their text content. It includes a graphical user interface (GUI) built using Python's Tkinter library.

## Objectives
- Detect spam emails automatically.
- Preprocess email text for machine learning.
- Convert text into numerical features using TF-IDF.
- Classify emails using the Multinomial Naive Bayes algorithm.
- Display predictions through a simple graphical interface.

## Technologies Used
- Python
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib
- Tkinter

## Features
- Enter an email subject and body.
- Predict whether an email is spam or legitimate.
- Display the result in a GUI.
- Clear the input fields and test another email.
- Evaluate the model using accuracy, precision, recall, and a confusion matrix.

## Model Performance
- **Test accuracy:** 95.89%
- **Dataset:** Apache SpamAssassin public email corpus

*Performance may vary on new emails, and some messages may be misclassified.*

## Project Files
- `spam_detector.py` — prepares the data, trains the model, and evaluates performance.
- `predict_email.py` — predicts the category of an email using the saved model.
- `email_gui.py` — provides the graphical user interface.
- `spam_email_model.pkl` — saved trained machine learning pipeline.
- `requirements.txt` — Python package dependencies.

## How to Run

1. Install Python.
2. Download or clone this repository.
3. Open a terminal in the project folder.
4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Launch the application:

   ```bash
   python email_gui.py
   ```

Ensure that `spam_email_model.pkl` is present in the project folder before running the GUI.

## Limitations
This project is an educational machine learning application. Its predictions are not guaranteed to be correct for every email.

## Author
Abigail
