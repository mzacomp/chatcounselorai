# app.py
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from dotenv import load_dotenv
import os
import openai
import numpy as np

#--OpenAIKey--#
load_dotenv()  # this loads variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")

# ---FastAPI and CORS Middleware--
from fastapi import FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# --- Load Artifacts and Path  ---
# Absolute path to ml_model/trained_model
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ml_model/trained_model"))

model = joblib.load(os.path.join(MODEL_DIR, "theme_multilabel_classifier.joblib"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
mlb = joblib.load(os.path.join(MODEL_DIR, "multilabel_binarizer.joblib"))





# --- Request Models ---
class PredictionRequest(BaseModel):
    input_text: str

class AdviceRequest(BaseModel):
    input_text: str
    predicted_themes: list[str]

# --- Utility ---
def predict_themes_top_k(text, top_k=3):
    processed = [text.lower().strip()]
    vectorized = vectorizer.transform(processed)
    probs = model.predict_proba(vectorized)[0]
    top_indices = np.argsort(probs)[-top_k:][::-1]
    return [mlb.classes_[i] for i in top_indices if probs[i] > 0]

def generate_advice(input_text, themes):
    prompt = f"""
You are an AI assistant supporting licensed mental health counselors. A counselor has provided a description of a patient's concerns and the system has predicted the following key themes: {', '.join(themes)}.

Based on this, generate clear, supportive, and evidence-informed advice that the counselor can use to help the patient.

Counselor Input:
{input_text}

Advice:
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )

    return response.choices[0].message["content"].strip()

# --- Routes ---

@app.post("/predict")
def predict_route(request: PredictionRequest):
    themes = predict_themes_top_k(request.input_text)
    return {"predicted_themes": themes}

@app.post("/generate-advice")
def advice_route(request: AdviceRequest):
    if not request.predicted_themes:
        raise HTTPException(status_code=400, detail="Themes required.")
    advice = generate_advice(request.input_text, request.predicted_themes)
    return {"advice": advice}
