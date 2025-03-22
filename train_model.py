import pandas as pd
import numpy as np
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score, make_scorer
from scipy.stats import uniform

# Load Dataset
file_path = "dt1.xlsx"  # Ensure this file is in the same directory
df = pd.read_excel(file_path)

# Split labels into a list
df['label_list'] = df['labels'].apply(lambda x: [label.strip() for label in x.split(',')])

# Count label frequency
label_counter = Counter([label for labels in df['label_list'] for label in labels])

# Define threshold for rare labels
rare_labels = {label for label, count in label_counter.items() if count < 3}

# Replace rare labels with 'Other'
df['processed_labels'] = df['label_list'].apply(lambda x: ['Other' if label in rare_labels else label for label in x])

# Multi-label binarization
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['processed_labels'])

# Text vectorization
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    sublinear_tf=True,
    min_df=2,
    max_df=0.85,
    stop_words='english'
)
X = vectorizer.fit_transform(df['text_snippet'])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and Hyperparameter Tuning
param_distributions = {
    'estimator__C': uniform(0.1, 10),
    'estimator__solver': ['liblinear', 'lbfgs', 'saga'],
    'estimator__penalty': ['l2', 'elasticnet'],
    'estimator__l1_ratio': uniform(0, 1)
}

base_model = LogisticRegression(max_iter=5000, class_weight='balanced')
multi_output_clf = MultiOutputClassifier(base_model)

random_search = RandomizedSearchCV(
    multi_output_clf,
    param_distributions,
    n_iter=20,
    cv=5,
    scoring=make_scorer(f1_score, average='micro'),
    verbose=2,
    random_state=42
)
random_search.fit(X_train, y_train)

# Best Model
best_model = random_search.best_estimator_

# Save the trained model and preprocessing tools
joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(mlb, "mlb.pkl")

print("\n✅ Model, vectorizer, and label binarizer saved successfully!")

# Final Evaluation
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=1))

# Prediction Function
def preprocess_text(text):
    return text.lower().strip()

def predict_labels(text):
    text_vectorized = vectorizer.transform([preprocess_text(text)])
    prediction = best_model.predict(text_vectorized)
    predicted_labels = mlb.inverse_transform(prediction)
    return predicted_labels[0] if predicted_labels else ["No labels detected"]

# Example Predictions
test_texts = [
    "How does your pricing compare to competitors?",
    "I'm interested in your product features",
    "What are your security measures?"
]

print("\nExample Predictions:")
for text in test_texts:
    print(f"Text: {text}")
    print(f"Predicted Labels: {predict_labels(text)}\n")
