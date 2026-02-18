import numpy as np
import scipy.sparse as sp
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Paths
ROOT = '.'
PROCESSED = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
X_train_sparse = sp.load_npz(os.path.join(PROCESSED, 'X_train.npz'))
X_test_sparse = sp.load_npz(os.path.join(PROCESSED, 'X_test.npz'))
y_train = np.load(os.path.join(PROCESSED, 'y_train.npy'))
y_test = np.load(os.path.join(PROCESSED, 'y_test.npy'))

if y_train.min() == 1:
    y_train = y_train - 1
    y_test = y_test - 1

def sparse_batch_generator(X, y, batch_size=128, shuffle=True):
    samples = X.shape[0]
    indices = np.arange(samples)
    while True:
        if shuffle: np.random.shuffle(indices)
        for start in range(0, samples, batch_size):
            end = min(start + batch_size, samples)
            batch_idx = indices[start:end]
            yield X[batch_idx].toarray(), y[batch_idx]

dim = X_train_sparse.shape[1]
model = Sequential([
    Input(shape=(dim,)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

BATCH_SIZE = 128
EPOCHS = 10 # Reduced for speed of evaluation

train_gen = sparse_batch_generator(X_train_sparse, y_train, batch_size=BATCH_SIZE)
val_gen = sparse_batch_generator(X_test_sparse, y_test, batch_size=BATCH_SIZE)

train_steps = X_train_sparse.shape[0] // BATCH_SIZE
val_steps = X_test_sparse.shape[0] // BATCH_SIZE

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

print("Starting training...")
model.fit(
    train_gen, steps_per_epoch=train_steps,
    epochs=EPOCHS, validation_data=val_gen, validation_steps=val_steps,
    callbacks=[early_stop], verbose=1
)

# Evaluate
print("\nEvaluating...")
X_test_dense = X_test_sparse.toarray()
y_pred = np.argmax(model.predict(X_test_dense), axis=1)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print(f"NN Accuracy: {acc:.4f}")
print(f"NN Macro-F1: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['World', 'Sports', 'Business', 'Sci/Tech']))

# Save
model.save(os.path.join(MODELS_DIR, 'nn_model.h5'))
print("Model saved to models/nn_model.h5")
