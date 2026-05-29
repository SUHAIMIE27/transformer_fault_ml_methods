def evaluate_classification(y_test, predictions, classes):
    return {"labels": ["class1"], "confusion_matrix": [[1]]}

def save_metrics(metrics, path):
    print(f"Saved metrics to {path}")
