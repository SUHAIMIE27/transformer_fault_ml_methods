# tests/test_train_model.py

from src.models.train_model import train_model

def test_train_model_runs():
    X_train = ["sample text"] * 5
    y_train = ["class1"] * 5
    model_path = "outputs/model/test_model.joblib"
    clf, vect = train_model(X_train, y_train, model_path)
    assert clf is not None
    assert vect is not None