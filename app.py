from flask import Flask, render_template, request
import joblib
import pandas as pd
from statistics import mode

# -----------------------------
# Load models and objects
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_index = joblib.load("symptom_index.pkl")

# Symptoms list
symptoms = list(symptom_index.keys())

# Flask app
app = Flask(__name__)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)

    for symptom in input_symptoms:
        symptom = symptom.strip()
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1

    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)[0]]
    svm_pred = encoder.classes_[svm_model.predict(input_df)[0]]

    final_pred = mode([rf_pred, nb_pred, svm_pred])

    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        symptoms_input = request.form["symptoms"]
        prediction = predict_disease(symptoms_input)
    return render_template("index.html", prediction=prediction, symptoms=symptoms)

if __name__ == "__main__":
    app.run(debug=True)
