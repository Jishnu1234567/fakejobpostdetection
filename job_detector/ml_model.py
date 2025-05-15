import joblib
import numpy as np

def load_model_and_vectorizer():
    model = joblib.load("ml_model/random_forest_model.pkl")
    vectorizer = joblib.load("ml_model/vectorizer.pkl")
    return model, vectorizer

def predict_job_post(text, model, vectorizer):
    vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(vectorized)[0]
    prediction = model.predict(vectorized)[0]
    confidence = max(probabilities)
    return prediction, confidence
