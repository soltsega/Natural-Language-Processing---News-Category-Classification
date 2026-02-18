# News Category Classification 📰

A professional, end-to-end NLP pipeline for classifying news articles into four major categories (**World**, **Sports**, **Business**, and **Sci/Tech**). This project leverages the AG News dataset and compares classical Machine Learning versus Deep Learning approaches.

---

## 🚀 Project Overview

This repository documents the development of a robust text classification system, following a structured 6-phase methodology. The pipeline covers everything from raw data ingestion to hyperparameter-tuned deployment artifacts.

### 📊 Performance Highlights
| Model | Accuracy | Macro-F1 | Inference Speed |
|-------|----------|----------|-----------------|
| **Tuned SGD (Linear SVM)** | **91.1%** | **91.1%** | **Fastest** |
| Logistic Regression | 90.8% | 90.8% | Fast |
| Neural Network (FFNN) | 90.8% | 90.8% | Moderate |
| XGBoost | 88.4% | 88.3% | Slow |

> **Verdict**: The **SGD Classifier** is the recommended model for production due to its optimal balance of predictive power and computational efficiency.

---

## 📂 Repository Structure

```text
.
├── data/               # Raw and processed CSVs/Arrays
├── docs/               # Detailed phase-by-phase reports
├── models/             # Exported .joblib and .h5 models
├── notebooks/          # Step-by-step experiment logs (01-04)
├── scripts/            # Automation scripts for preprocessing/training
├── src/                # Modular source code (logic/features)
└── requirements.txt    # Project dependencies
```

---

## 🛠️ The 6-Phase Pipeline

### Phase 1: Infrastructure & Data Ingestion
- Configured a modular workspace and robust data loading utility (`src/data_loader.py`).
- Integrated the **AG News** dataset (120k Train / 7.6k Test samples).

### Phase 2: Exploratory Data Analysis (EDA)
- Confirmed a **perfect class balance** (30,000 samples per category).
- Visualized top keywords via per-class Word Clouds and frequency histograms.

### Phase 3: Preprocessing Pipeline
- Implemented `clean_text` in `src/preprocessing.py`.
- Applied: Lowercasing, HTML/Noise removal, and **WordNet Lemmatization**.

### Phase 4: Feature Engineering (TF-IDF)
- Transformed text into a **50,000-dimensional sparse matrix**.
- Used **Unigrams + Bigrams** with `sublinear_tf` scaling to capture complex phrases like *"world cup"* or *"interest rates"*.

### Phase 5: Modeling & Hyperparameter Tuning
- Benchmarked LogReg, Random Forest, XGBoost, and SGD.
- Performed `RandomizedSearchCV` on the winning SGD model, achieving **91.1% Accuracy**.

### Phase 6: Deep Learning Benchmark
- Built a **Simple Feedforward Neural Network** in TensorFlow/Keras.
- Used a custom sparse-batch generator to handle high-dimensional input efficiently on standard hardware.

---

## 🔧 Installation & Setup

1. **Clone the Repo**:
   ```bash
   git clone https://github.com/soltsega/Natural-Language-Processing---News-Category-Classification.git
   cd Natural-Language-Processing---News-Category-Classification
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚦 Usage

- **Interactive Implementation**: Run the notebooks in chronological order (01 to 04).
- **Batch Preprocessing**:
  ```bash
  python scripts/preprocess_data.py
  ```
- **Inference**: Use the saved models in `models/` (see `best_model.joblib`).

---

## 📜 Findings & Design Decisions
- **Lemmatization vs Stemming**: Lemmatization was chosen for its ability to maintain semantic meaning, which proved critical for differentiating "Business" and "Sci/Tech".
- **Sparse Storage**: Given the ~99.95% sparsity of the TF-IDF matrix, all matrices are stored in `CSR` format, reducing storage needs by over 200x.
- **Model Selection**: While Neural Networks provide a modern approach, the high-dimensional sparse nature of TF-IDF features allows Linear SVMs (via SGD) to achieve equivalent accuracy with significantly lower latency.

---
**Developed with ❤️ by SolTsega**
