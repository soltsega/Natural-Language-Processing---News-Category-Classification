# Phase 6 Report: Deep Learning & Evaluation

## Overview

In the final phase of the project, we implemented a **Simple Feedforward Neural Network (FFNN)** using TensorFlow/Keras to serve as a deep learning benchmark against the classical machine learning models trained in Phase 5. All experiments are documented in [`notebooks/04_deep_learning.ipynb`](../notebooks/04_deep_learning.ipynb).

---

## Neural Network Architecture

To handle the high-dimensional TF-IDF input (50,000 features), we designed a compact yet effective architecture:

| Layer (type) | Units/Rate | Activation |
|--------------|------------|------------|
| Input        | 50,000     | -          |
| Dense (1)    | 128        | ReLU       |
| Dropout      | 0.5        | -          |
| Dense (2)    | 64         | ReLU       |
| Dropout      | 0.3        | -          |
| Dense (Out)  | 4          | Softmax    |

- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Regularization**: Dropout + Early Stopping

---

## 1. Neural Network Performance

The model was trained for 10 epochs using a sparse-to-dense data generator to manage memory constraints.

| Metric | Accuracy | Macro-F1 |
|--------|----------|----------|
| **Value** | **0.9084** | **0.9084** |

### Per-Class Performance (NN)

| Class | F1-Score |
|-------|----------|
| **World** | 0.91 |
| **Sports** | 0.96 |
| **Business** | 0.88 |
| **Sci/Tech** | 0.89 |

---

## 2. Final Multi-Model Comparison

We aggregated the results from all phases to identify the overall winner for this AG News classification task.

| Phase | Model | Accuracy | Macro-F1 | Complexity |
|-------|-------|----------|----------|------------|
| Phase 5 | **SGD (Linear SVM)** | **0.9109** | **0.9108** | **Low** (Fastest) |
| Phase 5 | Logistic Regression | 0.9085 | 0.9082 | Low |
| Phase 6 | **Neural Network** | 0.9084 | 0.9084 | Medium |
| Phase 5 | XGBoost | 0.8842 | 0.8835 | High |
| Phase 5 | Random Forest | 0.8715 | 0.8708 | Medium |

---

## Conclusions

1. **ML vs. DL**: For this specific dataset and feature set (TF-IDF), the **Classical ML (SGD/SVM)** approach slightly outperformed the Neural Network in both accuracy and training efficiency.
2. **Efficiency**: The SGD classifier trained in seconds, whereas the Neural Network required several minutes and significanly more computational resources (RAM/GPU).
3. **Robustness**: All top models (SGD, LR, NN) performed exceptionally well on the **Sports** category but struggled slightly more with the overlap between **Business** and **Sci/Tech**.

### Final Recommendation

The **Tuned SGD Classifier** is selected as the production-ready model for its superior efficiency and comparable (slightly better) accuracy on high-dimensional text data.

---

## Output Artifacts

- **NN Model**: [`models/nn_model.h5`](../models/nn_model.h5)
- **NN Confusion Matrix**: [`models/nn_confusion_matrix.png`](../models/nn_confusion_matrix.png)
- **Final Notebook**: [`notebooks/04_deep_learning.ipynb`](../notebooks/04_deep_learning.ipynb)
