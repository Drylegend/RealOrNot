# utils/reddit_utils.py

import joblib
import numpy as np

# Load model and vectorizer only once
_model = None
_vectorizer = None

def load_model():
    global _model, _vectorizer
    if _model is None or _vectorizer is None:
        _model = joblib.load("models/fake_news_model.pkl")
        _vectorizer = joblib.load("models/vectorizer.pkl")
    return _model, _vectorizer

def classify_reddit_title(title):
    model, vectorizer = load_model()
    vect = vectorizer.transform([title])
    pred = model.predict(vect)[0]
    raw_conf = model.decision_function(vect)[0]
    label = "Fake" if pred == 0 else "Real"
    conf_pct = round(1 / (1 + np.exp(-raw_conf)) * 100, 1)
    return label, conf_pct
