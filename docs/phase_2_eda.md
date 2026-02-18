# Phase 2 Report: Exploratory Data Analysis (EDA)

## Overview
Phase 2 focused on understanding the AG News dataset through statistical analysis and visualization. The goal was to identify class distributions, text characteristics, and category-specific keywords to inform the preprocessing and modeling strategy.

## 1. Dataset Statistics
- **Training Samples**: 120,000
- **Testing Samples**: 7,600
- **Features**: `Class Index` (Category), `Title`, `Description`.
- **Target Categories**: 
    1. World
    2. Sports
    3. Business
    4. Sci/Tech

## 2. Key Findings
### Class Distribution
The dataset is perfectly balanced. Each of the four categories constitutes exactly 25% of the total samples in the training set (30,000 articles per class). This eliminates the need for class weighting or oversampling.

### Text Length Analysis
- **Average Description Length**: ~31 words.
- **Median Description Length**: 30 words.
- The distribution is relatively standard with a slight right tail, suggesting most news summaries are concise.

### Keyword Insights
Using frequency analysis and word clouds, we identified distinct vocabularies:
- **World**: Focuses on geopolitical terms (Bush, Iraq, Palestine, Minister).
- **Sports**: High frequency of competitive terms (Olympic, Games, Win, League, Team).
- **Business**: Dominated by corporate and financial terms (Company, Inc, Stocks, Oil, Prices).
- **Sci/Tech**: characterized by technical and digital terms (Software, Internet, Microsoft, New, Space).

## 3. Preprocessing Implications
Based on the EDA:
- **Title + Description**: Combining these features is recommended to provide more context for the models.
- **Stopwords**: Common English stops should be removed to reduce noise.
- **Lemmatization**: Suggested to group variants like "Companies" and "Company" together.
- **TF-IDF**: High distinguishing words between classes suggest that TF-IDF with n-grams will be highly effective.

## 4. Completed Artifacts
- **Notebook**: `01_eda.ipynb` (Consolidated logic, plots, and word clouds).
- **Utilities**: `src/eda_utils.py` (Helper functions for visualization).

---
**Next Step**: Phase 3 - Preprocessing Pipeline (Tokenization, Cleaning, Lemmatization).
