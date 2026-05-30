# transformer_fault_ml_methods

## Project Overview
This project implements a **Transformer Fault Diagnosis ML Methods pipeline**.  
It loads a dataset of transformer-fault literature, preprocesses it, trains a classification model, evaluates results, and generates metrics and visualizations.

---

## Folder Structure
transformer_fault_ml_methods/
├── main.py # Orchestrates full pipeline
├── config.py # Project configuration (paths, settings)
├── data/
│   ├── raw/ # Original Excel dataset
│   │   └── relevant_transformer_fault_diagnosis.xlsx
│   └── processed_transformer_fault_records.csv
├── outputs/
│   ├── figures/ # Plots (category distribution, confusion matrix, top methods, publications by year)
│   ├── metrics/ # Metrics JSON and pipeline summary
│   └── model/ # Trained models
├── src/
│   ├── data/data_loader.py
│   ├── models/train_model.py
│   ├── preprocessing/preprocessing.py
│   ├── preprocessing/feature_engineering.py
│   ├── evaluation/evaluate.py
│   └── utils/
│       ├── reporting.py
│       └── visualization.py
├── tests/
│   └── test_train_model.py
├── README.md
├── DECOMPOSITION_REPORT.md
├── PRESENTATION_SCRIPT.md
└── requirements.txt

---

## Code Explanation

The project is structured modularly. Here's what each module does:

### 1. `main.py`
- Orchestrates the entire ML pipeline.
- Steps:
  1. Loads raw data using `DatasetLoader`.
  2. Preprocesses text columns with `TransformerFaultPreprocessor`.
  3. Trains the ML classifier using `MLCategoryClassifierTrainer`.
  4. Evaluates predictions with `evaluate_classification`.
  5. Generates plots and summary reports.
  6. Saves processed data, models, metrics, and visualizations.

### 2. `config.py`
- Defines paths for data, outputs, and models.
- Sets pipeline parameters (test size, random seed, text columns, target column).

### 3. `src/data/data_loader.py`
- `DatasetLoader` class: loads Excel dataset and validates required columns.
- `save_processed_data()` saves processed datasets as CSV.

### 4. `src/preprocessing/preprocessing.py`
- `TransformerFaultPreprocessor`:
  - Cleans text data.
  - Combines text columns into a single feature column for ML.
  - Handles missing values.

### 5. `src/models/train_model.py`
- `train_model()` trains a TF-IDF + LinearSVC pipeline.
- `MLCategoryClassifierTrainer`:
  - Splits data into train/test.
  - Fits classifier and predicts labels.
  - Returns model, predictions, and test labels.
- `save_model()` stores the trained model using joblib.

### 6. `src/evaluation/evaluate.py`
- `evaluate_classification()` calculates metrics (accuracy, confusion matrix).
- `save_metrics()` saves metrics as JSON.

### 7. `src/utils/reporting.py`
- `make_summary()` generates a textual summary of processed data and class distribution.
- `save_text_report()` writes the summary to a `.txt` file.

### 8. `src/utils/visualization.py`
- Functions to create plots:
  - `plot_category_distribution()` → bar chart of target classes.
  - `plot_confusion_matrix()` → shows classifier performance.
  - `plot_publications_by_year()` → line chart of publications over time.
  - `plot_top_methods()` → top ML methods used in literature.

### 9. `tests/`
- Contains unit tests for core modules:
  - `test_train_model.py` → ensures `train_model()` runs and outputs a classifier and vectorizer.
  - `test_main.py` → optional minimal tests for pipeline integrity.

---

## Pipeline Workflow

1. **Data Loading:** Reads raw Excel data and checks required columns.
2. **Preprocessing:** Cleans and combines text columns into a single feature column.
3. **Training:** Splits data, trains TF-IDF + LinearSVC classifier (or dummy classifier if insufficient classes).
4. **Evaluation:** Generates predictions, calculates metrics, saves JSON metrics.
5. **Visualization:** Plots results and saves figures.
6. **Reporting:** Generates a pipeline summary text file with class distributions.
7. **Saving:** Stores processed CSV, trained model, metrics, figures, and summary text.

---

## Installation & Setup

1. Extract the ZIP file.
2. Install dependencies:

```bash
pip install -r requirements.txt