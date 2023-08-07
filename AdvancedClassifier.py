import re
import string
import spacy
from spacy.lang.en import English
from sklearn.ensemble import GradientBoostingClassifier
from gensim.models import Word2Vec
from xgboost import XGBClassifier
from sklearn.svm import SVC
from transformers import BertTokenizer, BertModel
import torch

# Mapping of algorithm names to corresponding objects
ALGORITHMS = {
    "Gradient Boosting": GradientBoostingClassifier,
    "XGBoost": XGBClassifier,
    "SVM": SVC,
    "BERT": BertModel,
    # Add more algorithms as needed
}

# Mapping of vectorizer names to corresponding objects
VECTORIZERS = {
    "Word2Vec": Word2Vec,
    "BERT Tokenizer": BertTokenizer,
    # Add more vectorizers as needed
}

# Load SpaCy NLP model for English
nlp = spacy.load('en_core_web_sm')
nlp_available = True

def advanced_clean_text(text, lowercase=True, remove_special_chars=True, remove_digits=True, remove_punctuation=True, remove_stopwords=True):
    """
    Advanced clean the input text using SpaCy.

    Parameters:
        text (str): The input text to be cleaned.
        lowercase (bool, optional): Convert text to lowercase. Default is True.
        remove_special_chars (bool, optional): Remove special characters. Default is True.
        remove_digits (bool, optional): Remove digits. Default is True.
        remove_punctuation (bool, optional): Remove punctuation. Default is True.
        remove_stopwords (bool, optional): Remove stop words. Default is True.

    Returns:
        str: The cleaned text.
    """
    # Remove special characters
    if remove_special_chars:
        text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Use SpaCy for preprocessing if available
    if nlp_available:
        doc = nlp(text)

        # Convert to lowercase
        if lowercase:
            cleaned_text = ' '.join(token.lemma_.lower() for token in doc if not token.is_punct)
        else:
            cleaned_text = ' '.join(token.lemma_ for token in doc if not token.is_punct)

    else:
        raise ImportError("SpaCy is not available. Please make sure you have installed the 'en_core_web_sm' model from SpaCy.")

    return cleaned_text

def train_text_classifier(messages, labels, algorithm_name, vectorizer_name):
    """
    Train a text classifier using user-selected advanced algorithm and vectorizer.

    Parameters:
        messages (list): List of input text messages.
        labels (list): List of corresponding labels for the messages.
        algorithm_name (str): Name of the user-selected advanced classification algorithm.
        vectorizer_name (str): Name of the user-selected advanced text vectorizer.

    Returns:
        object: Trained classification model.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Invalid algorithm name. Supported algorithms are: {', '.join(ALGORITHMS.keys())}")

    if vectorizer_name not in VECTORIZERS:
        raise ValueError(f"Invalid vectorizer name. Supported vectorizers are: {', '.join(VECTORIZERS.keys())}")

    cleaned_texts = [clean_text(text) for text in messages]

    # Initialize the vectorizer based on user input
    vectorizer = VECTORIZERS[vectorizer_name]

    # Convert text to numerical features using the chosen vectorizer
    X = vectorizer(cleaned_texts)

    # Initialize the algorithm based on user input
    algorithm = ALGORITHMS[algorithm_name]

    # Train the model
    model = algorithm()
    model.fit(X, labels)

    return model, vectorizer
