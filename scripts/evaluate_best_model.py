import joblib
import numpy as np
import scipy.sparse as sp
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = '.'
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')

# Load data
X_train = sp.load_npz(os.path.join(PROCESSED, 'X_train.npz'))
y_train = np.load(os.path.join(PROCESSED, 'y_train.npy'))
X_test = sp.load_npz(os.path.join(PROCESSED, 'X_test.npz'))
y_test = np.load(os.path.join(PROCESSED, 'y_test.npy'))

if y_train.min() == 1:
    y_train = y_train - 1
    y_test = y_test - 1

class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

def evaluate(model_path, name):
    if not os.path.exists(model_path):
        return None
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return {
        'Model': name,
        'Accuracy': acc,
        'Macro-F1': f1,
        'Type': type(model).__name__
    }

# Evaluating best model
best_res = evaluate(os.path.join(MODELS_DIR, 'best_model.joblib'), 'Best Model (Tuned)')
if best_res:
    print("--- Best Model Metrics ---")
    print(f"Model: {best_res['Type']}")
    print(f"Accuracy: {best_res['Accuracy']:.4f}")
    print(f"Macro-F1: {best_res['Macro-F1']:.4f}")

# Classification report for best model
model = joblib.load(os.path.join(MODELS_DIR, 'best_model.joblib'))
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# We don't have the other baselines saved individually, but we can infer them from the notebook
# or just provide the best model metrics in the report.
