import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import os
from sklearn.metrics import accuracy_score, f1_score

ROOT = '.'
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')

model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'nn_model.h5'))
X_test = sp.load_npz(os.path.join(PROCESSED, 'X_test.npz')).toarray()
y_test = np.load(os.path.join(PROCESSED, 'y_test.npy'))

if y_test.min() == 1:
    y_test = y_test - 1

y_prob = model.predict(X_test)
y_pred = np.argmax(y_prob, axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"NN_ACC={acc:.4f}")
print(f"NN_F1={f1:.4f}")
