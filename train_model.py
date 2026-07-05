"""
SignPlay ASL — train a fast 36-class letter/digit classifier.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATASET_PATH = Path(__file__).with_name("dataset.csv")
MODEL_PATH = Path(__file__).with_name("model.pkl")
ENCODER_PATH = Path(__file__).with_name("scaler_encoder.pkl")


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Missing dataset at {DATASET_PATH}. Run data_collector.py first.")

    df = pd.read_csv(DATASET_PATH)
    if "label" not in df.columns:
        raise ValueError("dataset.csv must contain a 'label' column.")

    feature_cols = [col for col in df.columns if col.startswith("f")]
    if len(feature_cols) != 60:
        raise ValueError(f"Expected 60 feature columns, found {len(feature_cols)}.")

    x_data = df[feature_cols]
    y_data = df["label"].astype(str).str.upper()
    return x_data, y_data


def train_and_save() -> None:
    x_data, y_data = load_dataset()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_data)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification report:\n")
    print(report)


if __name__ == "__main__":
    train_and_save()