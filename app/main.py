from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

# =========================
# LOAD MODEL
# =========================
pipeline = joblib.load("sentiment_pipeline.pkl")
model = pipeline["model"]
glove_dict = pipeline["glove"]
tfidf = pipeline["tfidf"]  # <-- Load the TF-IDF vectorizer

def text_to_tfidf_glove(text, glove_dict, tfidf, vector_size=100):
    tokens = text.lower().split()
    vocab = tfidf.vocabulary_
    
    weighted_vectors = []
    weights =[]
    
    for token in tokens:
        if token in glove_dict and token in vocab:
            idx = vocab[token]
            weight = tfidf.idf_[idx]
            weighted_vectors.append(glove_dict[token] * weight)
            weights.append(weight)
            
    if len(weighted_vectors) > 0:
        vec = np.sum(weighted_vectors, axis=0) / np.sum(weights)
        return vec.reshape(1, -1)
    
    return np.zeros((1, vector_size))

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="BankFive Sentiment API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    # Use the new weighted function
    X_input = text_to_tfidf_glove(req.text, glove_dict, tfidf)

    prediction = model.predict(X_input)[0]
    classes = model.classes_

    idx = list(classes).index(prediction)
    confidence = float(probs[idx])

    return {
        "sentiment": prediction.lower(),
        "confidence": confidence
    }