
# Spam Message Detection Using Machine Learning
# B.Tech CSE - Fundamentals of AI Project

import pandas as pd
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1. Function to clean SMS messages
def clean_message(message):

    # Convert text to lowercase
    message = message.lower()

    # Remove punctuation
    message = message.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers
    message = re.sub(r'\d+', '', message)

    # Remove extra spaces
    message = re.sub(r'\s+', ' ', message).strip()

    return message


# 2. Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# 3. Rename columns
data = data.rename(columns={
    "v1": "label",
    "v2": "message"
})

# 4. Keep only required columns
data = data[["label", "message"]]

# 5. Remove missing values
data = data.dropna()

# 6. Clean messages
data["message"] = data["message"].apply(clean_message)

# 7. Convert labels into numbers
# ham = 0, spam = 1
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

data = data.dropna(subset=["label"])


# 8. Separate input and output
X = data["message"]
y = data["label"].astype(int)


# 9. Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 10. Convert messages into numerical features
vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 11. Create and train Naive Bayes model
model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# 12. Test the model
y_pred = model.predict(X_test_tfidf)


# 13. Display evaluation results
print("\n===== MODEL EVALUATION =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred))

print("Precision:",
      precision_score(y_test, y_pred, zero_division=0))

print("Recall:",
      recall_score(y_test, y_pred, zero_division=0))

print("F1-score:",
      f1_score(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 14. Display spam detector menu
while True:

    print("\n===== SPAM MESSAGE DETECTOR =====")
    print("1. Check a message")
    print("2. Exit")

    choice = input("Enter your choice: ")

    # 15. Check a new message
    if choice == "1":

        message = input("\nEnter SMS message: ")

        # Clean the message
        cleaned_message = clean_message(message)

        # Convert message into TF-IDF features
        message_tfidf = vectorizer.transform(
            [cleaned_message]
        )

        # Predict using trained model
        prediction = model.predict(message_tfidf)[0]

        # Display prediction
        if prediction == 1:
            print("\nPrediction: SPAM")
        else:
            print("\nPrediction: NOT SPAM")

    # 16. Exit application
    elif choice == "2":

        print("\nThank you for using Spam Message Detector!")
        break

    # 17. Invalid choice
    else:

        print("Invalid choice. Please try again.")
