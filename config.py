"""Project configuration values.

The assignment is designed to run in GitHub Codespaces. All paths are kept
relative to the repository root so the project works after cloning.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "relevant_transformer_fault_diagnosis.xlsx"
PROCESSED_DATA_PATH = DATA_DIR / "processed_transformer_fault_records.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model"
MODEL_PATH = MODEL_DIR / "tfidf_linear_svm_model.joblib"

TEXT_COLUMNS = ["Title", "Abstract", "Author Keywords", "Index Keywords"]
TARGET_COLUMN = "target_category"
RANDOM_STATE = 42
TEST_SIZE = 0.25
MIN_CLASS_COUNT = 5
