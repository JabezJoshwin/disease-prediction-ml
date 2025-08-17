import pickle
import pandas as pd

def load_model(path):
    return pickle.load(open(path, "rb"))

def preprocess_input(symptom_index, input_symptoms):
    input_symptoms = input_symptoms.split(",")
    input_data = [0] * len(symptom_index)
    for symptom in input_symptoms:
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
    return input_data
