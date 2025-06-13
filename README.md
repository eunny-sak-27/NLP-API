#  NLP API with FastAPI – Multi-label Classification & Entity Extraction

This project implements a **FastAPI** application that serves a **machine learning model** trained to perform **multi-label text classification** along with **entity extraction** using spaCy.

---

##  Features

- Multi-label text classification using a trained Logistic Regression model
- TF-IDF based text vectorization
- Label binarization using MultiLabelBinarizer
- Named Entity Recognition (NER) using a custom `extract_combined_entities` function
- Fast and scalable REST API using **FastAPI**
- Clean and structured responses in JSON

---

## 🗃️ Project Structure

├── app.py # FastAPI app
├── model.pkl # Trained ML model
├── vectorizer.pkl # Trained TF-IDF vectorizer
├── mlb.pkl # MultiLabelBinarizer instance
├── entity_extraction.py # Custom entity extraction function
├── dt1.xlsx # Original dataset
├── requirements.txt # Python dependencies
└── README.md
