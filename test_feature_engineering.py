"""Model selection and model training utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseClassifierTrainer(ABC):
    """Abstract trainer defining the common interface for classifiers."""

    @abstractmethod
    def prepare_training_data(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Return features and labels ready for training."""

    @abstractmethod
    def train(self, data: pd.DataFrame) -> dict[str, object]:
        """Train a model and return the fitted model plus test data."""


@dataclass
class MLCategoryClassifierTrainer(BaseClassifierTrainer):
    """Train a TF-IDF + Linear SVM classifier.

    The model predicts the dominant machine-learning-method category from each
    article's title, abstract, and keywords. Linear SVM is efficient for sparse text features.
    """

    text_column: str = "combined_text"
    target_column: str = "target_category"
    min_class_count: int = 5
    test_size: float = 0.25
    random_state: int = 42

    def prepare_training_data(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        usable = data.loc[data[self.target_column].ne("No Method Extracted")].copy()
        usable = usable.loc[usable[self.text_column].str.len().gt(0)]

        class_counts = usable[self.target_column].value_counts()
        valid_classes = class_counts[class_counts >= self.min_class_count].index
        usable = usable.loc[usable[self.target_column].isin(valid_classes)]

        if usable[self.target_column].nunique() < 2:
            raise ValueError("At least two target classes are required for classification.")

        return usable[self.text_column], usable[self.target_column]

    def train(self, data: pd.DataFrame) -> dict[str, object]:
        texts, labels = self.prepare_training_data(data)
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )

        model = Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=3000,
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words="english",
                    ),
                ),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        return {
            "model": model,
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "predictions": predictions,
            "classes": sorted(labels.unique().tolist()),
        }


def save_model(model: Pipeline, output_path: Path) -> None:
    """Persist the trained scikit-learn pipeline."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
