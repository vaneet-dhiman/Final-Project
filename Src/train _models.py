from pathlib import Path
import joblib
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "Models"
    / "student_performance_model.pkl"
)
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH =PROJECT_ROOT/"Models"/"student performance_model.pkl"
MODEL_PATH.parent.mkdir(parents=True ,exist_ok=True)

DATA_PATH = (
    PROJECT_ROOT
    / "Data"
    / "processed"
    / "student_performance_clean.csv"
)


def load_data():

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    return df


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


def create_preprocessor(X):

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

    numerical_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_transformer,
                numerical_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

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

    print(name)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["Fail", "Pass"]
        )
    )

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


def main():

    df = load_data()

    X, y = prepare_data(df)

    print("\nFeature shape:")
    print(X.shape)

    print("\nTarget distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    preprocessor = create_preprocessor(X)

    models = {
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

    results = []

    for name, classifier in models.items():

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
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
        if name=="Logistic Regression":
            joblib.dump(pipeline, MODEL_PATH)
            print(f"\nModel saved to: {MODEL_PATH}")

        result = evaluate_model(
            name,
            pipeline,
            X_test,
            y_test
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("MODEL COMPARISON")

    print(
        results_df
        .sort_values(
            "F1 Score",
            ascending=False
        )
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()