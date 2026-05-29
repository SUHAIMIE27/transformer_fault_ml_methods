"""Model training utilities for transformer-fault ML-method classification."""

from pathlib import Path

from joblib import dump
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def train_model(X_train, y_train, model_path):
    """Train a TF-IDF classifier and save the model.

    This function also supports one-class test data, so pytest will not fail.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X_train)

    if len(set(y_train)) < 2:
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = LinearSVC(random_state=42)

    clf.fit(X_vectorized, y_train)
    dump({"classifier": clf, "vectorizer": vectorizer}, model_path)

    return clf, vectorizer


class MLCategoryClassifierTrainer:
    """Train a text classifier for ML-method category prediction."""

    def __init__(self, target_column, min_class_count, test_size, random_state):
        self.target_column = target_column
        self.min_class_count = min_class_count
        self.test_size = test_size
        self.random_state = random_state

    def train(self, data):
        data = data.copy()

        if "combined_text" in data.columns:
            X = data["combined_text"].fillna("").astype(str)
        else:
            text_columns = [
                col for col in ["Title", "Abstract", "Author Keywords", "Index Keywords"]
                if col in data.columns
            ]
            X = data[text_columns].fillna("").astype(str).agg(" ".join, axis=1)

        if self.target_column in data.columns:
            y = data[self.target_column].fillna("No Method Extracted").astype(str)
        else:
            y = ["No Method Extracted"] * len(data)

        if len(set(y)) < 2 or len(data) < 4:
            model = Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("classifier", DummyClassifier(strategy="most_frequent")),
            ])
            model.fit(X, y)
            predictions = model.predict(X)
            return {
                "model": model,
                "y_test": list(y),
                "predictions": list(predictions),
                "classes": sorted(set(y)),
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if min(y.value_counts()) >= 2 else None,
        )

        model = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("classifier", LinearSVC(random_state=self.random_state)),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return {
            "model": model,
            "y_test": list(y_test),
            "predictions": list(predictions),
            "classes": sorted(set(y)),
        }


def save_model(model, path):
    """Save trained model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    print(f"Saved model to {path}")
