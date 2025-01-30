import joblib

def predict_news(text, model_path, tfidf_path):
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)

    text_processed = preprocess_text(text)
    text_vectorized = tfidf.transform([text_processed])
    prediction = model.predict(text_vectorized)
    return prediction[0]
