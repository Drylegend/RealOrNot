# utils/text_utils.py

import joblib
import numpy as np

MODEL_PATH = "models/fake_news_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

def load_text_model():
    """
    Load and return (model, vectorizer).
    """
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer

def predict_text(text: str, model, vectorizer):
    """
    Returns a dict:
      {
        "pred": 0|1,
        "raw_conf": float,        # decision_function output
        "conf_prob": float,       # sigmoid(raw_conf)
        "label": "❗ FAKE NEWS"|"✅ REAL NEWS",
        "conf_pct": float         # abs(raw_conf)*100
      }
    """
    X = vectorizer.transform([text])
    raw = model.decision_function(X)[0]
    pred = int(model.predict(X)[0])
    conf_prob = 1 / (1 + np.exp(-raw))
    label = "✅ REAL NEWS" if pred == 1 else "❗ FAKE NEWS"
    return {
        "pred": pred,
        "raw_conf": raw,
        "conf_prob": conf_prob,
        "label": label,
        "conf_pct": abs(raw) * 100
    }
