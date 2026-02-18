# Phase 5: Modeling & Hyperparameter Tuning

**Notebook**: [`notebooks/03_modeling.ipynb`](file:///c:/Users/My%20Device/Desktop/News-Category-Classification-E/notebooks/03_modeling.ipynb)  
**Date**: 2026-02-18  
**Input**: TF-IDF sparse matrices from Phase 4 (`data/processed/`)

---

## Overview

Three Scikit-Learn classifiers were trained on the 50k-feature TF-IDF matrix, then the best-performing model was tuned with `RandomizedSearchCV` (5-fold stratified CV, 20 iterations, scoring = macro-F1).

---

## Baseline Configurations

| Model | Key Hyperparameters |
|-------|---------------------|
| Logistic Regression | `C=5`, `solver=lbfgs`, `multi_class=multinomial`, `max_iter=1000` |
| SGD (Linear SVM) | `loss=modified_huber`, `max_iter=100` |
| Random Forest | `n_estimators=200`, `n_jobs=-1` |

---

## Hyperparameter Search Space

| Model | Parameters Searched |
|-------|---------------------|
| Logistic Regression | `C` ∈ LogUniform(0.01, 100), `max_iter` ∈ {500, 1000, 2000} |
| SGD | `alpha` ∈ LogUniform(1e-5, 0.1), `max_iter` ∈ {100, 200, 500}, `loss` ∈ {modified_huber, hinge} |
| Random Forest | `n_estimators` ∈ Randint(100, 500), `max_depth` ∈ {None, 20, 40}, `min_samples_split` ∈ Randint(2, 10) |

---

## Output Artifacts

| File | Description |
|------|-------------|
| `models/best_model.joblib` | Serialized best estimator |
| `models/confusion_matrix.png` | Confusion matrix heatmap |
| `models/per_class_f1.png` | Per-class F1: baseline vs tuned |
| `models/model_comparison.png` | Accuracy & Macro-F1 bar chart |

---

## Design Decisions

- **No CLI script** — all logic lives in the notebook for conciseness and reproducibility.
- **RandomizedSearchCV** preferred over GridSearchCV for speed on large sparse matrices.
- **Macro-F1** used as the primary metric (balanced across 4 classes).
- **Best model auto-selected** at runtime — whichever baseline wins drives the tuning step.
