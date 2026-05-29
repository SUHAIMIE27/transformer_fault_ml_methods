# Decomposition Report

## Project Title
Transformer Fault Diagnosis ML Methods Classification

## Public Source Used as Coding Reference
The coursework asks for a publicly available machine-learning or data-science code source and then a decomposition into a structured repository. This project uses the public scikit-learn text-classification example, **Classification of text documents using sparse features**, as the coding reference for the text-classification workflow. The example demonstrates TF-IDF text features and classifiers suitable for sparse text matrices.

Source URL: https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html

## Dataset Source Used in This Repository
The data source used for this assignment is the provided Excel workbook:

`data/raw/relevant_transformer_fault_diagnosis.xlsx`

The workbook contains Scopus records about power-transformer and dissolved-gas-analysis fault diagnosis. Important columns include article title, year, abstract, author keywords, index keywords, and a JSON-like extraction column describing machine-learning methods mentioned in each abstract.

## Machine-Learning Problem
The raw `Screening decision` column contains only one class (`Keep`), so it is not suitable for a meaningful classification model. Instead, the project derives a new target label called `target_category` from the first extracted machine-learning method/category in each record.

The model predicts the dominant machine-learning-method category from the combined title, abstract, and keyword text.

## Decomposed Repository Structure

```text
transformer_fault_ml_methods/
├── main.py
├── config.py
├── requirements.txt
├── data/
│   └── raw/
│       └── relevant_transformer_fault_diagnosis.xlsx
├── src/
│   ├── data/
│   │   └── data_loader.py
│   ├── preprocessing/
│   │   └── preprocessing.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── train_model.py
│   ├── evaluation/
│   │   └── evaluate.py
│   └── utils/
│       ├── visualization.py
│       └── reporting.py
├── tests/
└── docs/
```

## Module Explanation

### `main.py`
Runs the whole workflow: loading, preprocessing, model training, evaluation, visualization, and output saving.

### `config.py`
Stores file paths and reusable constants, making the project easier to modify without changing many scripts.

### `src/data/data_loader.py`
Contains `DatasetLoader`, which loads the Excel file and checks that all required columns exist.

### `src/preprocessing/preprocessing.py`
Contains `MethodExtractionParser` and `TransformerFaultPreprocessor`. These classes parse the JSON-like extraction column, clean missing text, combine textual fields, and create the derived `target_category` label.

### `src/features/feature_engineering.py`
Contains an abstract base class `TextFeatureBuilder` and an inherited class `TfidfFeatureBuilder`. This demonstrates abstraction and inheritance while also showing how text is converted into numeric machine-learning features.

### `src/models/train_model.py`
Contains `BaseClassifierTrainer` and `MLCategoryClassifierTrainer`. The inherited trainer builds a TF-IDF + Linear SVM model.

### `src/evaluation/evaluate.py`
Computes accuracy, macro-F1, weighted-F1, a classification report, and a confusion matrix.

### `src/utils/visualization.py`
Creates saved figures for publication trends, method-category distribution, top extracted methods, and the confusion matrix.

### `tests/`
Contains unit tests for data loading, preprocessing, feature engineering, model training, and evaluation.

## OOP Concepts for Presentation

- **Class:** `DatasetLoader`, `TransformerFaultPreprocessor`, `MLCategoryClassifierTrainer`.
- **Object:** An object is created in `main.py`, such as `loader = DatasetLoader(RAW_DATA_PATH)`.
- **Abstraction:** `TextFeatureBuilder` and `BaseClassifierTrainer` define required methods without exposing unnecessary implementation details.
- **Inheritance:** `TfidfFeatureBuilder` inherits from `TextFeatureBuilder`; `MLCategoryClassifierTrainer` inherits from `BaseClassifierTrainer`.
- **Encapsulation:** Each class keeps related behavior together. For example, `DatasetLoader` handles loading and column validation inside one class.

## How to Run

```bash
pip install -r requirements.txt
python main.py
pytest
```

## Expected Outputs

After running `python main.py`, the project creates:

- `data/processed_transformer_fault_records.csv`
- `outputs/metrics/classification_metrics.json`
- `outputs/metrics/pipeline_summary.txt`
- `outputs/model/tfidf_linear_svm_model.joblib`
- `outputs/figures/publications_by_year.png`
- `outputs/figures/category_distribution.png`
- `outputs/figures/top_methods.png`
- `outputs/figures/confusion_matrix.png`
