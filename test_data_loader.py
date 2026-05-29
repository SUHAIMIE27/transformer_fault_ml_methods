"""Feature-engineering classes for text-based machine learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TextFeatureBuilder(ABC):
    """Abstract base class for any text feature builder.

    This class demonstrates abstraction. Concrete classes must implement both
    `fit_transform` and `transform`.
    """

    @abstractmethod
    def fit_transform(self, texts: list[str]) -> csr_matrix:
        """Fit the feature builder and transform training text."""

    @abstractmethod
    def transform(self, texts: list[str]) -> csr_matrix:
        """Transform unseen text using the fitted feature builder."""


@dataclass
class TfidfFeatureBuilder(TextFeatureBuilder):
    """Create TF-IDF features from article titles, abstracts, and keywords."""

    max_features: int = 2500
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    stop_words: str = "english"

    def __post_init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stop_words=self.stop_words,
        )

    def fit_transform(self, texts: list[str]) -> csr_matrix:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list[str]) -> csr_matrix:
        return self.vectorizer.transform(texts)
