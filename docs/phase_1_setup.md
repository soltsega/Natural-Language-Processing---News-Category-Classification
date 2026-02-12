# Phase 1 Report: Project Initialization & Data Setup

## Overview
Phase 1 established the foundational structure for the News Category Classification project. This phase focused on setting up a robust development environment, maximizing reproducibility through version control and dependency management, and securing the AG News dataset for downstream tasks.

## 1. Project Architecture
The project follows a modular structure to separate data, source code, scripts, and notebooks:

```
News-Category-Classification-E/
├── data/
│   ├── raw/             # Original AG News dataset (train.csv, test.csv)
│   └── processed/       # Placeholder for cleaned/tokenized data
├── docs/                # Project documentation
├── models/              # Directory for saving trained models
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── scripts/             # Utility scripts (download, verify)
│   ├── download_data.py # Script to fetch dataset from remote source
│   ├── run_eda.py       # (Planned) EDA execution script
│   └── verify_env.py    # Environment verification script
├── src/                 # Source code modules
│   ├── data_loader.py   # Data loading logic
│   ├── preprocessing.py # Text cleaning functions
│   └── train.py         # Model training pipeline
├── .gitignore           # Git ignore rules
├── README.md            # Project entry point
└── requirements.txt     # Python dependencies
```

## 2. Environment Setup
- **Virtual Environment**: A Python `venv` was created to isolate dependencies.
- **Dependencies**: Key libraries installed via `requirements.txt`:
    - `pandas`, `numpy`: specific versions for data manipulation.
    - `scikit-learn`: For machine learning models.
    - `nltk`: For text preprocessing (stopwords, lemmatization).
    - `matplotlib`, `seaborn`: For visualization.
    - `requests`: For data downloading.

## 3. Data Acquisition
- **Dataset**: AG News (News Articles)
- **Source**: Downloaded from a stable GitHub mirror via `scripts/download_data.py`.
- **Files**:
    - `train.csv`: Training set with 120,000 samples.
    - `test.csv`: Testing set with 7,600 samples.
    - `classes.txt`: Class labels (World, Sports, Business, Sci/Tech).
- **Verification**: `src/data_loader.py` implemented to consistently load and parse the CSV files, handling potential header variances.

## 4. Key Artifacts Created
| Artifact | Description |
| :--- | :--- |
| `scripts/download_data.py` | Robust script to download data, handling network errors and using `requests` or `urllib`. |
| `src/data_loader.py` | utility function `load_data()` to return train/test DataFrames. |
| `src/preprocessing.py` | Initial implementation of `clean_text` function for normalization. |
| `src/verify_env.py` | Script to check if all required packages are importable. |

## 5. Next Steps (Phase 2)
The next phase will focus on **Exploratory Data Analysis (EDA)** to:
- Visualize class distribution.
- Analyze text length and vocabulary.
- Generate word clouds for each category.
