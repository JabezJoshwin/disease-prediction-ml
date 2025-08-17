import os
import pandas as pd
import pickle
from statistics import mode

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "../models")
DATA_DIR = os.path.join(BASE_DIR, "../data")

rf_model = pickle.load(open(os.path.join(MODELS_DIR, "rf_model.pkl"), "rb"))
nb_model = pickle.load(open(os.path.join(MODELS_DIR, "nb_model.pkl"), "rb"))
svm_model = pickle.load(open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb"))
encoder = pickle.load(open(os.path.join(MODELS_DIR, "encoder.pkl"), "rb"))

data = pd.read_csv(os.path.join(DATA_DIR, "improved_disease_dataset.csv"))
symptoms = data.columns.values[:-1]
symptom_index = {symptom: idx for idx, symptom in enumerate(symptoms)}

def predict_disease(input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
    input_df = pd.DataFrame([input_data], columns=symptoms)

    rf_pred = encoder.classes_[rf_model.predict(input_df)[0]]
    nb_pred = encoder.classes_[nb_model.predict(input_df)]   # Added  to get scalar prediction
    svm_pred = encoder.classes_[svm_model.predict(input_df)] # Added  similarly

    final_pred = mode([rf_pred, nb_pred, svm_pred])
    return {
        "Random Forest Prediction": rf_pred,
        "Naive Bayes Prediction": nb_pred,
        "SVM Prediction": svm_pred,
        "Final Prediction": final_pred
    }

# Example usage
if __name__ == "__main__":
    print(predict_disease("skin_rash,fever,headache"))
