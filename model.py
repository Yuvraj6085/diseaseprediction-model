import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# -----------------------------
# Load dataset
# -----------------------------
url = "https://raw.githubusercontent.com/Am1rreza/Disease-Prediction/main/improved_disease_dataset.csv"
df = pd.read_csv(url)

# Encode target column
encoder = LabelEncoder()
df['disease'] = encoder.fit_transform(df['disease'])

X = df.drop("disease", axis=1)
y = df['disease']

# Symptom index mapping
symptom_index = {col: idx for idx, col in enumerate(X.columns)}

# -----------------------------
# Train models
# -----------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
nb_model = GaussianNB()
svm_model = SVC(probability=True, kernel="linear")

rf_model.fit(X, y)
nb_model.fit(X, y)
svm_model.fit(X, y)

# -----------------------------
# Save models and objects
# -----------------------------
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(nb_model, "nb_model.pkl")
joblib.dump(svm_model, "svm_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(symptom_index, "symptom_index.pkl")

print("✅ Models and encoders saved successfully!")
