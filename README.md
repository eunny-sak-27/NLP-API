#  NLP API with FastAPI – Multi-label Classification & Entity Extraction

## 📌 Project Description

This project implements an end-to-end **NLP microservice** using **FastAPI** that performs:

1. **Multi-label text classification** on sales/marketing call snippets using a trained Logistic Regression model.  
2. **Domain-specific entity extraction** using a hybrid approach:  
   - A **custom dictionary lookup** from `domain_knowledge.json` (e.g., competitors, pricing terms, product features).  
   - A **spaCy-based Named Entity Recognition (NER)** pipeline for general-purpose extraction.

The system exposes a **RESTful API** that accepts a JSON input containing a text snippet and returns:
- The predicted **classification labels** (e.g., `Objection`, `Pricing Discussion`, `Security`, `Competition`)
- Extracted **domain-specific entities** using both techniques

All components are **containerized using Docker**, enabling seamless deployment to cloud platforms like **Render** or local execution.


---

##  Features

- Multi-label text classification using a trained Logistic Regression model
- TF-IDF based text vectorization
- Label binarization using MultiLabelBinarizer
- Named Entity Recognition (NER) using a custom `extract_combined_entities` function
- Fast and scalable REST API using **FastAPI**
- Clean and structured responses in JSON

---

##  Project Structure

- `app.py` – FastAPI application
- `model.pkl` – Trained machine learning model (Logistic Regression)
- `vectorizer.pkl` – Trained TF-IDF vectorizer
- `mlb.pkl` – MultiLabelBinarizer instance used for multi-label binarization
- `entity_extraction.py` – Custom entity extraction function (`extract_combined_entities`)
- `dt1.xlsx` – Original labeled dataset used for training
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation


---

## ⚙️ Setup Instructions

### 1.  Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---

### 3. Ensure these files exist in the directory

- `model.pkl`
- `vectorizer.pkl`
- `mlb.pkl` 
- `entity_extraction.py: must contain a function extract_combined_entities(text: str) -> dict`

---

### 4. Run the FastAPI app
```bash
uvicorn app:app --reload
```
By default, it will run on: http://localhost:8000

---
### 4. API Endpoints
POST /predict
Description: Classifies a piece of text and extracts entities.
Request Body:
```bash
{
  "text": "What are your pricing and security policies?"
}
```
Response:
```bash
{
  "text": "What are your pricing and security policies?",
  "predicted_labels": ["Pricing", "Security"],
  "extracted_entities": {
    "ORG": ["YourCompany"],
    "POLICY": ["pricing policy", "security policy"]
  }
}
```
---

### GET /
Simple health check route.

Response:
```bash
{
  "message": "FastAPI NLP Model with Classification & Entity Extraction"
}
```
## 🔧 How It Works

1. Input text is received via a `POST` request to `/predict`
2. The text is transformed using a trained `TfidfVectorizer`
3. The machine learning model (`MultiOutputClassifier`) predicts multiple labels
4. Predicted labels are decoded using `MultiLabelBinarizer`
5. Custom entity extraction is performed using `extract_combined_entities`
6. The final result is returned as a structured JSON object

---

### Testing the API (Swagger UI)
Visit: http://localhost:8000/docs

Swagger UI is automatically available with FastAPI for testing all endpoints.

---

## 🚀 Deployment on Render

This FastAPI project is deployed live using [Render](https://render.com) with containerized deployment.

### Deployment Steps

### Hosting on Render (Step-by-Step)

1. Push your project to a GitHub repository
2. Go to [https://render.com](https://render.com) and log in
3. Click **"New → Web Service"**
4. Connect your GitHub repo
5. Fill in the details:
   - **Environment**: Python 3
   - **Build Command**:  
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:  
     ```bash
     uvicorn app:app --host 0.0.0.0 --port 10000
     ```
   - **Port**: 10000 (or leave as default if unspecified)
6. Click **Create Web Service**
7. Wait for Render to build and deploy your app
8. Once deployed, test with:
   ```bash
   curl -X POST https://nlp-service-wuq5.onrender.com/predict -H "Content-Type: application/json" -d "{\"text\": \"Can you explain your pricing model and refund policy for enterprise customers?\"}"
   ```
   RESULT
![image](https://github.com/user-attachments/assets/a56ce317-df11-4a47-8833-531b89a64e13)

---
###    Built By:
**Saketh Eunny**
