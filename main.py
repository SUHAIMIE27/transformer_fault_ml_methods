"""Run the full transformer-fault ML-method classification pipeline."""

from __future__ import annotations

from config import (
    FIGURE_DIR,
    METRICS_DIR,
    MIN_CLASS_COUNT,
    MODEL_PATH,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    TEXT_COLUMNS,
)
from src.data.data_loader import DatasetLoader, save_processed_data
from src.evaluation.evaluate import evaluate_classification, save_metrics
from src.models.train_model import MLCategoryClassifierTrainer, save_model
from src.preprocessing.preprocessing import TransformerFaultPreprocessor
from src.utils.reporting import make_summary, save_text_report
from src.utils.visualization import (
    plot_category_distribution,
    plot_confusion_matrix,
    plot_publications_by_year,
    plot_top_methods,
)


def main() -> None:
    """Execute data loading, preprocessing, training, evaluation, and outputs."""
    loader = DatasetLoader(RAW_DATA_PATH)
    raw_data = loader.load()

    preprocessor = TransformerFaultPreprocessor(TEXT_COLUMNS)
    processed_data = preprocessor.transform(raw_data)
    save_processed_data(processed_data, PROCESSED_DATA_PATH)

    trainer = MLCategoryClassifierTrainer(
        target_column=TARGET_COLUMN,
        min_class_count=MIN_CLASS_COUNT,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    training_result = trainer.train(processed_data)

    metrics = evaluate_classification(
        training_result["y_test"],
        training_result["predictions"],
        training_result["classes"],
    )
    save_metrics(metrics, METRICS_DIR / "classification_metrics.json")
    save_model(training_result["model"], MODEL_PATH)

    plot_publications_by_year(processed_data, FIGURE_DIR / "publications_by_year.png")
    plot_category_distribution(processed_data, FIGURE_DIR / "category_distribution.png")
    plot_top_methods(processed_data, FIGURE_DIR / "top_methods.png")
    plot_confusion_matrix(metrics["labels"], metrics["confusion_matrix"], FIGURE_DIR / "confusion_matrix.png")

    summary = make_summary(processed_data)
    save_text_report(summary, METRICS_DIR / "pipeline_summary.txt")

    print(summary)
    print("\nSaved processed data, model, metrics, and figures in the outputs/ and data/ folders.")


if __name__ == "__main__":
    main()
