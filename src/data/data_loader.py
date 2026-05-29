# Transformer Fault Diagnosis ML Methods - Pipeline Summary
# =============================================================
# Total records processed: 441
# Publication year range: 1997 to 2026
# Records with at least one extracted ML method: 366
#
# Target category distribution:
# - General AI / ML: 133
# - Neural Network / Deep Learning: 112
# - No Method Extracted: 75
# - Supervised Learning: 69
# - Ensemble Learning: 19
# - Optimization: 16
# - Statistical / Feature Methods: 7
# - Other / Hybrid Methods: 6
# - Fuzzy / Expert Systems: 2
# - Unsupervised Learning: 2

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
    """Load and validate the raw Excel dataset."""

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