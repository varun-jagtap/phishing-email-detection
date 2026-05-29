import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PHISHING_KEYWORDS = [
    "urgent",
    "verify",
    "suspended",
    "account",
    "password",
    "login",
    "confirm",
    "otp",
    "alert",
    "action required",
    "pay",
    "invoice",
    "refund",
    "security",
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    features_df = df.copy()

    def url_count(text: str) -> int:
        return len(URL_PATTERN.findall(text))

    def keyword_count(text: str) -> int:
        lower_text = text.lower()
        return sum(lower_text.count(keyword) for keyword in PHISHING_KEYWORDS)

    features_df["url_count"] = features_df["email_text"].apply(url_count)
    features_df["keyword_count"] = features_df["email_text"].apply(keyword_count)
    return features_df


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1), "email_text"),
            (
                "extra",
                Pipeline([("scale", StandardScaler(with_mean=False))]),
                ["url_count", "keyword_count"],
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("features", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    expected_columns = {"email_text", "label"}
    if set(df.columns) != expected_columns:
        raise ValueError("Dataset must contain exactly 'email_text' and 'label' columns")

    valid_labels = {"Phishing", "Safe"}
    labels = set(df["label"].unique())
    if not labels.issubset(valid_labels):
        raise ValueError("Label values must be either 'Phishing' or 'Safe'")

    return df


def train_and_evaluate(dataset_path: Path) -> Pipeline:
    df = extract_features(load_dataset(dataset_path))

    X = df[["email_text", "url_count", "keyword_count"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    cm = confusion_matrix(y_test, predictions, labels=["Phishing", "Safe"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Phishing", "Actual: Safe"],
        columns=["Predicted: Phishing", "Predicted: Safe"],
    )
    print("Confusion Matrix:")
    print(cm_df.to_string())

    return model


def classify_email(model: Pipeline, email_text: str) -> str:
    data = pd.DataFrame(
        {
            "email_text": [email_text],
            "url_count": [len(URL_PATTERN.findall(email_text))],
            "keyword_count": [
                sum(email_text.lower().count(keyword) for keyword in PHISHING_KEYWORDS)
            ],
        }
    )
    return str(model.predict(data)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phishing email detection model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/emails.csv"),
        help="Path to CSV dataset with email_text,label columns",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Optional email text to classify after training",
    )

    args = parser.parse_args()
    model = train_and_evaluate(args.data)

    if args.predict:
        prediction = classify_email(model, args.predict)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
