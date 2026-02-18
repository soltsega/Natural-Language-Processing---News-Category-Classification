# Phase 3 Report: Preprocessing Pipeline

## Overview
Phase 3 focused on transforming the raw AG News text data into a clean, normalized form ready for feature engineering and modeling. The preprocessing pipeline was implemented entirely within `notebooks/PreprocessingPipline.ipynb` and applied to both the training and test sets.

## 1. Dataset Inputs
- **Training Set**: `data/raw/train.csv` — 120,000 samples
- **Test Set**: `data/raw/test.csv` — 7,600 samples
- **Columns used**: `Description` (primary text field for cleaning)

## 2. Preprocessing Steps

The `clean_text` function applies the following transformations in sequence:

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | **Lowercasing** | Normalize case for consistent tokenization |
| 2 | **HTML Tag Removal** | Strip any `<tag>` artifacts from the raw text |
| 3 | **Agency Tag Removal** | Remove press markers like `Reuters -`, `(AP)`, `(AFP)` |
| 4 | **Special Character & Number Removal** | Keep only alphabetic characters |
| 5 | **Stopword Removal** | Remove common English stopwords (NLTK corpus) |
| 6 | **Lemmatization** | Reduce words to their base form using `WordNetLemmatizer` |

### Why Lemmatization over Stemming?
Lemmatization was preferred because it produces valid dictionary words (e.g., `authorities` → `authority`, `prices` → `price`), preserving semantic meaning. Stemming can produce non-words (e.g., `running` → `runn`) which may hurt model interpretability.

## 3. Sample Transformation

| Raw Description | Cleaned Description |
|----------------|---------------------|
| `Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.` | `shortsellers wall street dwindlingband ultracynics seeing green` |
| `Reuters - Authorities have halted oil export flows from the main pipeline...` | `authority halted oil exportflows main pipeline...` |
| `AFP - Tearaway world oil prices, toppling records and straining wallets...` | `tearaway world oil price toppling record straining wallet...` |

## 4. NLTK Resources Used
- `stopwords` — English stopword list
- `wordnet` — Lexical database for lemmatization
- `omw-1.4` — Open Multilingual Wordnet (required by `wordnet`)

## 5. Output Artifacts
Cleaned datasets were saved to the `data/processed/` directory:

| File | Size | Rows |
|------|------|------|
| `clean_train.csv` | ~43.8 MB | 120,000 |
| `clean_test.csv` | ~2.8 MB | 7,600 |

Each file contains the original columns (`Class Index`, `Title`, `Description`) plus the new `Clean_Description` column.

## 6. Execution Time
- Training set preprocessing: ~2 minutes
- Test set preprocessing: ~5 seconds
- Total notebook execution: ~3 minutes

## 7. Implications for Phase 4
- The `Clean_Description` column is the primary input for TF-IDF vectorization.
- Stopword removal reduces vocabulary size, improving TF-IDF efficiency.
- Lemmatization groups word variants, boosting term frequency signals for rare but meaningful words.

---
**Next Step**: Phase 4 — Feature Engineering (TF-IDF Vectorization with n-grams).
