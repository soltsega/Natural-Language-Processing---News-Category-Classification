# Phase 4 Report: Feature Engineering

## Overview

Phase 4 converts the cleaned AG News text into numeric feature matrices using **TF-IDF vectorization** with unigrams and bigrams. All work is contained in [`notebooks/02_feature_engineering.ipynb`](../notebooks/02_feature_engineering.ipynb).

---

## Goal

Transform preprocessed text data (`Clean_Description`) into sparse numeric matrices that machine learning models can consume, while preserving discriminative signal across the four AG News categories:

| Class Index | Label    |
|-------------|----------|
| 1           | World    |
| 2           | Sports   |
| 3           | Business |
| 4           | Sci/Tech |

---

## Inputs

| File | Rows | Description |
|------|------|-------------|
| `data/processed/clean_train.csv` | 120,000 | Preprocessed training set |
| `data/processed/clean_test.csv`  | 7,600   | Preprocessed test set |

The key column used for vectorization is `Clean_Description` — the lemmatized, stop-word-free version of each article's description.

---

## 1. Vectorizer Configuration

The `FeatureEngineer` class (defined in `src/features.py`) wraps scikit-learn's `TfidfVectorizer` with the following settings:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `ngram_range` | `(1, 2)` | Captures single words **and** two-word phrases (bigrams) |
| `max_features` | `50,000` | Caps vocabulary size to control memory usage |
| `sublinear_tf` | `True` | Applies `log(1 + TF)` to dampen very frequent terms |
| `min_df` | `2` | Drops terms appearing in only one document (noise reduction) |
| `strip_accents` | `unicode` | Normalises accented characters for consistency |

> **Why `sublinear_tf=True`?** A word appearing 100× in a document is not 100× more important than one appearing 10×. Log-scaling compresses this effect and improves downstream model performance.

---

## 2. Class Distribution (Training Set)

The dataset is **perfectly balanced** — 30,000 samples per class:

```
Business    30,000
Sci/Tech    30,000
Sports      30,000
World       30,000
```

This means no class-imbalance handling is needed at the feature engineering stage.

---

## 3. Feature Matrix Statistics

| Split | Shape | Sparsity |
|-------|-------|----------|
| Train | `(120,000 × 50,000)` | ~99.95% |
| Test  | `(7,600 × 50,000)`   | ~99.95% |

The matrices are stored in **scipy sparse format** (CSR), which is memory-efficient for high-sparsity data.

---

## 4. Vocabulary Breakdown

Of the 50,000 features in the vocabulary:

| Token Type | Count  |
|------------|--------|
| Unigrams   | 17,758 |
| Bigrams    | 32,242 |

Bigrams make up the majority of the vocabulary, capturing meaningful two-word phrases like `"stock market"`, `"world cup"`, and `"space shuttle"`.

---

## 5. Top Discriminative Terms (by IDF)

Terms with the **highest IDF scores** are the rarest and most discriminative across the corpus. A sample of the top 20:

| Term | IDF Score |
|------|-----------|
| `trafficshield` | 11.60 |
| `ltmeta` | 11.60 |
| `gt ltmeta` | 11.60 |
| `triano` | 11.09 |
| `shopping cart` | 11.09 |
| `link popularity` | 10.90 |
| `cranberry` | 10.90 |
| `cray` | 10.90 |

> **Note:** Some high-IDF terms are HTML artifacts or very rare proper nouns. These are retained since they appear in at least 2 documents (`min_df=2`) and may still carry signal.

---

## 6. Top TF-IDF Terms per Class

For each class, the **mean TF-IDF score** across all documents in that class was computed, and the top 15 terms were visualised. The chart is saved to:

```
docs/phase4_top_terms.png
```

These per-class top terms confirm that the vectorizer captures class-specific vocabulary effectively:
- **World**: geopolitical terms, country names
- **Sports**: athlete names, sport-specific vocabulary
- **Business**: financial and economic terms
- **Sci/Tech**: technology brands, scientific terminology

---

## 7. Output Artifacts

| File | Format | Description |
|------|--------|-------------|
| `models/tfidf_vectorizer.joblib` | joblib | Fitted TF-IDF vectorizer (for inference) |
| `data/processed/X_train.npz` | scipy sparse | Sparse train feature matrix `(120,000 × 50,000)` |
| `data/processed/X_test.npz`  | scipy sparse | Sparse test feature matrix `(7,600 × 50,000)` |
| `data/processed/y_train.npy` | numpy array | Train labels `(120,000,)` |
| `data/processed/y_test.npy`  | numpy array | Test labels `(7,600,)` |

> **Important:** The vectorizer is fitted **only on the training set** and then applied to the test set — preventing data leakage.

---

## 8. Sanity Check

A quick round-trip sanity check was performed after export:

1. Reload `tfidf_vectorizer.joblib` from disk
2. Re-transform a sample of training text
3. Verify the output shape and sparsity match the original matrices

All checks passed ✅

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use `Clean_Description` only (not `Title`) | Descriptions are longer and richer; combining both would require careful weighting |
| `max_features=50,000` | Balances vocabulary coverage with memory/speed constraints |
| Bigrams included | Captures compound concepts (e.g., `"interest rate"`, `"world cup"`) that unigrams miss |
| Sparse matrix format | ~99.95% sparsity makes dense storage impractical; scipy sparse is ~200× more efficient |

---

## Next Step

**Phase 5 — Modeling & Hyperparameter Tuning**

The exported feature matrices (`X_train.npz`, `X_test.npz`) and labels (`y_train.npy`, `y_test.npy`) will be used to train and evaluate classification models (e.g., Logistic Regression, Linear SVM, Multinomial Naive Bayes).
