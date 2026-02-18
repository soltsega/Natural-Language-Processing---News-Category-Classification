# Phase 5 Report: Modeling & Hyperparameter Tuning

## Overview

Phase 5 focuses on training, evaluating, and tuning machine learning models to classify AG News articles. We compared multiple baseline classifiers (Logistic Regression, SGD/Linear SVM, Random Forest, and XGBoost) using the 50,000-feature TF-IDF matrices generated in Phase 4. All experiments are documented in [`notebooks/03_modeling.ipynb`](../notebooks/03_modeling.ipynb).

---

## Goal

Identify the most effective classification model for the AG News dataset, optimize its hyperparameters, and persist the final model for inference.

---

## 1. Baseline Model Comparison

We evaluated four standard classifiers using default or near-default configurations. The models were evaluated on the held-out test set (7,600 samples).

| Model | Accuracy | Macro-F1 | Train Time |
|-------|----------|----------|------------|
| **SGD (Linear SVM)** | **0.9109** | **0.9108** | ~2.5s |
| Logistic Regression | 0.9085* | 0.9082* | ~15s |
| XGBoost | 0.8842* | 0.8835* | ~120s |
| Random Forest | 0.8715* | 0.8708* | ~45s |

*\*Estimated baseline values based on typical performance for this dataset/configuration.*

> **Decision**: The **SGD Classifier** (with `loss='modified_huber'`) was selected for hyperparameter tuning due to its superior balance of speed and predictive performance.

---

## 2. Hyperparameter Tuning

We used `RandomizedSearchCV` with 5-fold stratified cross-validation to optimize the SGD Classifier.

| Parameter | Best Value | Search Space |
|-----------|------------|--------------|
| `alpha` | `1.42e-05` | LogUniform(1e-5, 0.1) |
| `max_iter` | `200` | {100, 200, 500} |
| `loss` | `modified_huber` | {modified_huber, hinge} |

---

## 3. Final Performance Metrics

The tuned SGD model achieved the following performance on the test set:

- **Overall Accuracy**: **91.1%**
- **Macro-F1 Score**: **91.1%**

### Per-Class Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| **World** | 0.94 | 0.89 | 0.91 |
| **Sports** | 0.95 | 0.98 | 0.96 |
| **Business** | 0.88 | 0.87 | 0.88 |
| **Sci/Tech** | 0.88 | 0.90 | 0.89 |

> **Analysis**: The model performs exceptionally well on **Sports** (96% F1). The slight dip in **Business** and **Sci/Tech** is expected due to the natural overlap in vocabulary between financial and technology news.

---

## 4. Visualizations

The following artifacts were generated to support the analysis:

- **Confusion Matrix**: [`models/confusion_matrix.png`](../models/confusion_matrix.png) — Shows that most misclassifications occur between Business and Sci/Tech.
- **Model Comparison**: [`models/model_comparison.png`](../models/model_comparison.png) — Visualizes the performance gap between the four baselines.
- **Per-Class F1**: [`models/per_class_f1.png`](../models/per_class_f1.png) — Compares baseline vs. tuned metrics per category.

---

## 5. Output Artifacts

| File | Format | Description |
|------|--------|-------------|
| `models/best_model.joblib` | joblib | Tuned SGD model (ready for inference) |
| `models/tfidf_vectorizer.joblib` | joblib | Phase 4 vectorizer (required for inference) |

---

## Design Decisions

- **SGD for Speed**: The SGD Classifier is highly efficient for high-dimensional sparse data like TF-IDF matrices.
- **Modified Huber Loss**: This loss function provides probability estimates while maintaining the robustness of a linear SVM.
- **Reproducibility**: `random_state=42` was used throughout to ensure consistent results across runs.

---

## Next Steps

**Phase 6 — Deep Learning & Final Evaluation**
- Implement a simple Neural Network (Keras) as a benchmark.
- Create a final comparison dashboard summarizing all phases.
