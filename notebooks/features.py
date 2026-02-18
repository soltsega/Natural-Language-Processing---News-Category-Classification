"""
features.py — TF-IDF Feature Engineering for AG News Classification
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureEngineer:
    """
    Wraps TfidfVectorizer with project-standard settings:
      - unigrams + bigrams  (ngram_range=(1, 2))
      - max 50,000 features
      - sublinear TF scaling
      - minimum document frequency of 2
    """

    def __init__(self, max_features: int = 50_000):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word",
        )

    def fit_transform(self, texts):
        """Fit on training texts and return sparse feature matrix."""
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        """Transform new texts using the already-fitted vectorizer."""
        return self.vectorizer.transform(texts)

    def save(self, path: str):
        """Persist the fitted vectorizer to disk."""
        joblib.dump(self.vectorizer, path)
        print(f"Vectorizer saved → {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureEngineer":
        """Load a previously saved vectorizer from disk."""
        obj = cls.__new__(cls)
        obj.vectorizer = joblib.load(path)
        return obj
