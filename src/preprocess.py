import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return " ".join(tokens)

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data['cleaned_text'] = data['text'].apply(preprocess_text)
    return data
