# News Category Classification

A comprehensive NLP pipeline for classifying news articles into four major categories: World, Sports, Business, and Sci/Tech.

## Project Overview
This project demonstrates an end-to-end machine learning workflow, from Exploratory Data Analysis (EDA) and text preprocessing to feature engineering and multi-model evaluation.

### Key Results
| Model | Accuracy | Macro-F1 | Verdict |
|-------|----------|----------|---------|
| **SGD Classifier (Tuned)** | **91.1%** | **91.1%** | **Recommended (Best Speed/Accuracy)** |
| Neural Network | 90.8% | 90.8% | Benchmarked |
| Logistic Regression | 90.8% | 90.8% | Strong Baseline |

## Structure
- `data/`: Raw and processed datasets.
- `notebooks/`: 
    - `01_eda.ipynb`: Data exploration.
    - `02_feature_engineering.ipynb`: TF-IDF vectorization.
    - `03_modeling.ipynb`: Classical ML (SGD, RF, XGBoost).
    - `04_deep_learning.ipynb`: Neural Network implementation.
- `src/`: Core logic for preprocessing and features.
- `models/`: Trained model artifacts (`.joblib`, `.h5`).
- `docs/`: Phase-by-phase reports and documentation.

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Preprocess data: `python scripts/preprocess_data.py`
2. Run notebooks or train scripts to reproduce modeling results.
