import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import pickle  # For saving models

# Load dataset
data = pd.read_csv('./data/improved_disease_dataset.csv')

encoder = LabelEncoder()
data["disease"] = encoder.fit_transform(data["disease"])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

if 'gender' in X_resampled.columns:
    le = LabelEncoder()
    X_resampled['gender'] = le.fit_transform(X_resampled['gender'])

X_resampled = X_resampled.fillna(0)
if len(y_resampled.shape) > 1:
    y_resampled = y_resampled.values.ravel()

# Train and save models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_resampled, y_resampled)
pickle.dump(rf_model, open("./models/rf_model.pkl", "wb"))

nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
pickle.dump(nb_model, open("./models/nb_model.pkl", "wb"))

svm_model = SVC()
svm_model.fit(X_resampled, y_resampled)
pickle.dump(svm_model, open("./models/svm_model.pkl", "wb"))

# Save encoder
pickle.dump(encoder, open("./models/encoder.pkl", "wb"))
