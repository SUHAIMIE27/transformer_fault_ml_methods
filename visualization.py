"""Preprocessing and cleaning logic for the transformer-fault dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class MethodExtractionParser:
    """Parse the JSON-like extraction column in the source spreadsheet."""

    extraction_column: str = "output abstract extraction"

    def parse(self, value: Any) -> dict[str, Any]:
        """Parse one extraction cell.

        Invalid or empty cells are returned as a safe empty structure rather than
        crashing the full pipeline.
        """
        if pd.isna(value):
            return {"machine_learning_methods": [], "xai_methods": [], "main_results": []}

        text = str(value).strip()
        if not text:
            return {"machine_learning_methods": [], "xai_methods": [], "main_results": []}

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = self._fallback_parse(text)

        if not isinstance(parsed, dict):
            return {"machine_learning_methods": [], "xai_methods": [], "main_results": []}
        parsed.setdefault("machine_learning_methods", [])
        parsed.setdefault("xai_methods", [])
        parsed.setdefault("main_results", [])
        return parsed

    @staticmethod
    def _fallback_parse(text: str) -> dict[str, Any]:
        """Fallback parser for mildly malformed extraction text."""
        method_names = re.findall(r'"method_name"\s*:\s*"([^"]+)"', text)
        categories = re.findall(r'"category"\s*:\s*"([^"]+)"', text)
        methods = []
        for index, method_name in enumerate(method_names):
            category = categories[index] if index < len(categories) else "unknown"
            methods.append({"method_name": method_name, "category": category})
        return {"machine_learning_methods": methods, "xai_methods": [], "main_results": []}


def normalize_category(category: str | None, method_name: str | None = None) -> str:
    """Convert many detailed method labels into explainable target classes."""
    combined = f"{category or ''} {method_name or ''}".lower()

    if any(term in combined for term in ["cnn", "lstm", "rnn", "gru", "ann", "neural", "deep learning", "autoencoder"]):
        return "Neural Network / Deep Learning"
    if any(term in combined for term in ["ensemble", "random forest", "xgboost", "catboost", "adaboost", "boosting", "bagging"]):
        return "Ensemble Learning"
    if any(term in combined for term in ["svm", "support vector", "knn", "nearest", "decision tree", "naive bayes", "logistic", "supervised", "classification", "regression"]):
        return "Supervised Learning"
    if any(term in combined for term in ["clustering", "k-means", "unsupervised", "som", "self-organizing"]):
        return "Unsupervised Learning"
    if any(term in combined for term in ["optimization", "metaheuristic", "genetic", "particle swarm", "pso", "gwo", "whale", "ant colony"]):
        return "Optimization"
    if any(term in combined for term in ["feature", "pca", "dimensionality", "statistical", "correlation", "bayesian"]):
        return "Statistical / Feature Methods"
    if any(term in combined for term in ["fuzzy", "expert system", "dga interpretation"]):
        return "Fuzzy / Expert Systems"
    if any(term in combined for term in ["machine learning", "artificial intelligence", "general ml", "general ai"]):
        return "General AI / ML"
    return "Other / Hybrid Methods"


@dataclass
class TransformerFaultPreprocessor:
    """Clean source columns and create labels for the ML pipeline."""

    text_columns: list[str]
    parser: MethodExtractionParser = field(default_factory=MethodExtractionParser)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a processed copy of the raw dataset."""
        processed = data.copy()
        processed["Year"] = pd.to_numeric(processed["Year"], errors="coerce")

        for column in self.text_columns:
            if column not in processed.columns:
                processed[column] = ""
            processed[column] = processed[column].fillna("").astype(str)

        processed["combined_text"] = processed[self.text_columns].agg(" ".join, axis=1)
        processed["combined_text"] = processed["combined_text"].str.replace(r"\s+", " ", regex=True).str.strip()

        parsed_records = processed[self.parser.extraction_column].apply(self.parser.parse)
        processed["method_names"] = parsed_records.apply(self._extract_method_names)
        processed["raw_categories"] = parsed_records.apply(self._extract_categories)
        processed["method_count"] = processed["method_names"].apply(lambda value: len(value.split("; ")) if value else 0)
        processed["dominant_method"] = parsed_records.apply(self._first_method_name)
        processed["dominant_raw_category"] = parsed_records.apply(self._first_category)
        processed["target_category"] = processed.apply(
            lambda row: normalize_category(row["dominant_raw_category"], row["dominant_method"]), axis=1
        )

        processed.loc[processed["dominant_method"].eq(""), "target_category"] = "No Method Extracted"
        return processed

    @staticmethod
    def _methods(record: dict[str, Any]) -> list[dict[str, Any]]:
        methods = record.get("machine_learning_methods", [])
        return methods if isinstance(methods, list) else []

    def _extract_method_names(self, record: dict[str, Any]) -> str:
        return "; ".join(str(item.get("method_name", "")).strip() for item in self._methods(record) if item.get("method_name"))

    def _extract_categories(self, record: dict[str, Any]) -> str:
        return "; ".join(str(item.get("category", "")).strip() for item in self._methods(record) if item.get("category"))

    def _first_method_name(self, record: dict[str, Any]) -> str:
        methods = self._methods(record)
        if not methods:
            return ""
        return str(methods[0].get("method_name", "")).strip()

    def _first_category(self, record: dict[str, Any]) -> str:
        methods = self._methods(record)
        if not methods:
            return ""
        return str(methods[0].get("category", "")).strip()
