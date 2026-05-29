"""Data loading utilities for the transformer-fault literature dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_COLUMNS = {
    "Title",
    "Year",
    "Abstract",
    "Author Keywords",
    "Index Keywords",
    "output abstract extraction",
}


@dataclass
class DatasetLoader:
    """Load and validate the raw Excel dataset.

    Parameters
    ----------
    file_path:
        Path to the raw Excel workbook supplied with this repository.
    required_columns:
        Column names that must exist before the pipeline can continue.
    """

    file_path: Path
    required_columns: Iterable[str] = tuple(REQUIRED_COLUMNS)

    def load(self) -> pd.DataFrame:
        """Return the dataset as a pandas DataFrame after validation."""
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Raw data file was not found: {self.file_path}. "
                "Place the Excel file in data/raw/ and run again."
            )

        data = pd.read_excel(self.file_path)
        self._validate_columns(data)
        return data

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Raise a helpful error if important columns are missing."""
        missing = set(self.required_columns).difference(data.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required column(s): {missing_list}")


def save_processed_data(data: pd.DataFrame, output_path: Path) -> None:
    """Save processed data as CSV so the output is easy to inspect."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
