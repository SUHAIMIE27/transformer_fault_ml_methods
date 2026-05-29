"""Evaluation utilities for the text classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def evaluate_classification(y_true, y_pred, labels: list[str]) -> dict[str, Any]:
    """Return standard evaluation metrics for a multi-class classifier."""
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def save_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    """Write evaluation metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
