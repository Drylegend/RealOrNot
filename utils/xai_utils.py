# utils/xai_utils.py

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

def explain_fake_news(text, model_path="models/fake_news_model.pkl", vect_path="models/vectorizer.pkl", num_features=10):
    """
    Returns a LIME TextExplanation object for the given text.
    """
    # Load model & vectorizer
    model = joblib.load(model_path)
    vectorizer = joblib.load(vect_path)

    # Set up explainer
    explainer = LimeTextExplainer(class_names=["Fake", "Real"])

    # Build a pseudo-probability function around decision_function
    def predict_proba(texts):
        X = vectorizer.transform(texts)
        scores = model.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))
        return np.vstack([1 - probs, probs]).T

    # Generate explanation
    explanation = explainer.explain_instance(text, predict_proba, num_features=num_features)
    return explanation
