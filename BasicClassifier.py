import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def clean_text(text):
    # Basic text cleaning: remove special characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def train_text_classifier(messages, labels, algorithm):

    # Perform basic text cleaning
    cleaned_texts = [clean_text(text) for text in messages]
    
    # Create the vectorizer (TfidfVectorizer in this example)
    vectorizer = TfidfVectorizer()
    
    # Convert text to numerical features using the vectorizer
    X = vectorizer.fit_transform(cleaned_texts)
    
    # Select the chosen algorithm for classification
    if algorithm == "Naive Bayes":
        model = MultinomialNB()
    elif algorithm == "SVM":
        model = SVC(kernel='linear', C=1.0)
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()
    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
    else:
        raise ValueError("Invalid algorithm choice. Supported algorithms are: Naive Bayes, SVM, Random Forest, and Logistic Regression.")
    
    # Train the model
    model.fit(X, labels)
    
    return model, vectorizer

def predict_text_classification(text_to_predict, trained_model, vectorizer):
    # Clean the input text
    cleaned_texts = [clean_text(text) for text in texts_to_predict]
    
    # Vectorize the cleaned text using the provided vectorizer
    X_new = vectorizer.transform(cleaned_text)
    
    # Make prediction using the provided trained model
    prediction = trained_model.predict(X_new)
    
    return prediction
