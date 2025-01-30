from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load model and TF-IDF vectorizer
model = joblib.load('models/naive_bayes_model.pkl')
tfidf = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text_processed = " ".join(text.lower().split())
    text_vectorized = tfidf.transform([text_processed])
    prediction = model.predict(text_vectorized)[0]
    return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
