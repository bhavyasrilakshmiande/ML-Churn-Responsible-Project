"""Beginner-friendly baselines for the customer churn training dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path(__file__).with_name("customer-churn-training.csv")
TARGET = "churned"
ID_COLUMN = "customer_id"


def print_metrics(name, y_true, predictions, probabilities=None):
    """Print the requested classification metrics for one model."""
    print(f"\n{name} metrics")
    print(f"Accuracy:  {accuracy_score(y_true, predictions):.3f}")
    print(f"Precision: {precision_score(y_true, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y_true, predictions, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y_true, predictions, zero_division=0):.3f}")
    print("Classification report:")
    print(classification_report(y_true, predictions, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted; labels 0, 1):")
    print(confusion_matrix(y_true, predictions, labels=[0, 1]))

    if probabilities is None:
        print("Calibration: unavailable because this model does not provide probabilities.")
    else:
        print(f"Brier score: {brier_score_loss(y_true, probabilities):.3f}")
        print(
            "Calibration warning: this estimate is unreliable with only the small "
            "test set in this dataset."
        )


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find {DATA_FILE.name} beside this script.")

    data = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {data.shape}")
    print(f"Column names: {list(data.columns)}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nMissing values by column:")
    print(data.isna().sum())
    print(f"\nDuplicate rows: {data.duplicated().sum()}")
    print("\nData types:")
    print(data.dtypes)
    print("\nDescriptive statistics for numerical columns:")
    print(data.select_dtypes(include="number").describe())

    target_counts = data[TARGET].value_counts().sort_index()
    print("\nChurned target distribution (0=non-churn, 1=churn):")
    print(target_counts)
    print("Churn percentage: {:.1f}%".format(data[TARGET].mean() * 100))
    print("Non-churn percentage: {:.1f}%".format((1 - data[TARGET].mean()) * 100))

    feature_columns = [column for column in data.columns if column not in {TARGET, ID_COLUMN}]
    categorical_features = data[feature_columns].select_dtypes(
        include=["str", "category", "bool"]
    ).columns.tolist()
    numerical_features = data[feature_columns].select_dtypes(
        include="number"
    ).columns.tolist()
    print(f"\nCategorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    print(f"Excluded identifier from model features: {ID_COLUMN}")

    # The target is removed before feature construction. The identifier is also
    # removed because an ID is not a meaningful customer behavior feature.
    X = data[feature_columns]
    y = data[TARGET]
    leakage_candidates = [
        column
        for column in feature_columns
        if "churn" in column.lower() or "target" in column.lower()
    ]
    print("\nTarget leakage check:")
    print(f"Target column excluded from X: {TARGET not in X.columns}")
    print(f"Identifier excluded from X: {ID_COLUMN not in X.columns}")
    print(f"Name-based leakage candidates requiring review: {leakage_candidates or 'None detected'}")

    if y.nunique() < 2:
        raise ValueError("At least two target classes are required for this comparison.")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two records for a safe split.")

    # Stratification keeps both classes represented in the small test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"\nTraining rows: {len(X_train)}; testing rows: {len(X_test)}")
    print("Test target distribution:")
    print(y_test.value_counts().sort_index())

    majority_model = DummyClassifier(strategy="most_frequent")
    majority_model.fit(X_train, y_train)
    majority_predictions = majority_model.predict(X_test)
    print_metrics("Majority-class non-ML baseline", y_test, majority_predictions)

    transformers = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numerical_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    logistic_model = Pipeline(
        steps=[
            ("preprocess", transformers),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    logistic_model.fit(X_train, y_train)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_probabilities = logistic_model.predict_proba(X_test)[:, 1]
    print_metrics(
        "Logistic Regression baseline",
        y_test,
        logistic_predictions,
        logistic_probabilities,
    )

    print("\nSimple error analysis for Logistic Regression:")
    error_table = X_test.copy()
    error_table["actual_churned"] = y_test
    error_table["predicted_churned"] = logistic_predictions
    error_table["error"] = error_table["actual_churned"] != error_table["predicted_churned"]
    print(error_table[error_table["error"]].to_string())
    print(f"Total test errors: {int(error_table['error'].sum())} of {len(error_table)}")

    if "plan_type" in data.columns:
        print("\nErrors by plan_type (descriptive only):")
        print(
            error_table.groupby("plan_type", observed=True)
            .agg(records=("error", "size"), errors=("error", "sum"))
        )
        print(
            "Warning: group-level fairness comparisons are unreliable with so few "
            "records, especially for small plan_type groups."
        )

    # The matrix is also plotted for a convenient visual check in an interactive
    # VS Code session. The printed matrix above remains usable in a text terminal.
    ConfusionMatrixDisplay.from_predictions(
        y_test, logistic_predictions, labels=[0, 1], display_labels=["No churn", "Churn"]
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()