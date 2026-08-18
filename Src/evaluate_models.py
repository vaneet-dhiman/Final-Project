from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "Outputs"
    / "evaluation"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


def load_data():

    return pd.read_csv(DATA_PATH)


def prepare_data(df):

    features = [
        "age",
        "gender",
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours",
        "internet_access",
        "parental_education",
        "family_income",
        "extracurricular",
        "study_environment"
    ]

    X = df[features]

    y = df["performance"]

    return X, y


def create_preprocessor():

    numerical_features = [
        "age",
        "study_hours",
        "attendance_percentage",
        "previous_score",
        "assignment_score",
        "midterm_score",
        "absences",
        "sleep_hours"
    ]

    categorical_features = [
        "gender",
        "internet_access",
        "parental_education",
        "family_income",
        "extracurricular",
        "study_environment"
    ]

    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_features
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            )
        ]
    )


def create_models():

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42
        )
    }


def evaluate_models(X_train, X_test, y_train, y_test):

    models = create_models()

    results = {}

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    create_preprocessor()
                ),
                (
                    "classifier",
                    classifier
                )
            ]
        )

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(X_test)

        probabilities = pipeline.predict_proba(
            X_test
        )[:, list(
            pipeline.classes_
        ).index("Fail")]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label="Fail"
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="Fail"
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="Fail"
        )

        auc = roc_auc_score(
            (y_test == "Fail").astype(int),
            probabilities
        )

        results[name] = {
            "model": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }

    return results


def save_confusion_matrices(results, y_test):

    for name, result in results.items():

        matrix = confusion_matrix(
            y_test,
            result["predictions"],
            labels=["Fail", "Pass"]
        )

        plt.figure(figsize=(6, 5))

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Fail", "Pass"],
            yticklabels=["Fail", "Pass"]
        )

        plt.title(
            f"{name} - Confusion Matrix"
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plt.tight_layout()

        filename = (
            name.lower()
            .replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            OUTPUT_PATH / filename
        )

        plt.close()


def save_roc_curve(results, y_test):

    actual = (
        y_test == "Fail"
    ).astype(int)

    plt.figure(figsize=(8, 6))

    for name, result in results.items():

        false_positive_rate, true_positive_rate, _ = roc_curve(
            actual,
            result["probabilities"]
        )

        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC = {result['auc']:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH / "roc_curve.png"
    )

    plt.close()


def save_model_comparison(results):

    rows = []

    for name, result in results.items():

        rows.append(
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1 Score": result["f1"],
                "ROC-AUC": result["auc"]
            }
        )

    comparison = pd.DataFrame(rows)

    comparison = comparison.sort_values(
        "F1 Score",
        ascending=False
    )

    print("MODEL EVALUATION")

    print(
        comparison.round(4).to_string(
            index=False
        )
    )

    comparison.to_csv(
        OUTPUT_PATH / "model_comparison.csv",
        index=False
    )

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]

    for metric in metrics:

        plt.figure(figsize=(8, 5))

        sns.barplot(
            data=comparison,
            x="Model",
            y=metric
        )

        plt.title(
            f"Model Comparison - {metric}"
        )

        plt.xticks(
            rotation=15
        )

        plt.tight_layout()

        filename = (
            metric.lower()
            .replace(" ", "_")
            + "_comparison.png"
        )

        plt.savefig(
            OUTPUT_PATH / filename
        )

        plt.close()


def main():

    df = load_data()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test
    )

    save_confusion_matrices(
        results,
        y_test
    )

    save_roc_curve(
        results,
        y_test
    )

    save_model_comparison(
        results
    )

    print("\nEvaluation completed successfully.")

    print(
        f"Results saved to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()