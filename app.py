from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
import re
import spacy
from entity_extraction import extract_combined_entities  # Import Task 2 functions

# Initialize FastAPI
app = FastAPI()

# Load trained ML model from Task 1
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
mlb = joblib.load("mlb.pkl")

# Define request format
class InputText(BaseModel):
    text: str

# Function to classify text (Task 1)
def classify_text(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    predicted_labels = mlb.inverse_transform(prediction)
    return predicted_labels[0] if predicted_labels else ["No labels detected"]

# API Endpoint: Classify & Extract Entities
@app.post("/predict")
def predict(data: InputText):
    text = data.text

    # Get classification labels (Task 1)
    predicted_labels = classify_text(text)

    # Get extracted entities (Task 2)
    extracted_entities = extract_combined_entities(text)

    return {
        "text": text,
        "predicted_labels": predicted_labels,
        "extracted_entities": extracted_entities
    }

# Home route
@app.get("/")
def home():
    return {"message": "FastAPI NLP Model with Classification & Entity Extraction"}
