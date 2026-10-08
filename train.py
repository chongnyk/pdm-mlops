"""Train candidate models, track them in MLflow, register the best one as @production."""
import inspect
import json
import os

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import FEATURES, TARGET, make_data


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "machine-failure"
MODEL_NAME = "machine-failure-model"

THRESHOLD = 0.3
MIN_PR_AUC = 0.7
SELECT_BY = "roc_auc"

MONITORING_REFERENCE_PATH = "monitoring_reference.json"
PSI_BINS = 10

TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

SAVE_OPTIONS = {}
if "skops_trusted_types" in inspect.signature(
    mlflow.sklearn.log_model
).parameters:
    SAVE_OPTIONS["skops_trusted_types"] = TRUSTED_TYPES


def check_todos():
    missing = [
        name
        for name, value in [
            ("THRESHOLD (TODO 2a)", THRESHOLD),
            ("MIN_PR_AUC (TODO 2b)", MIN_PR_AUC),
            ("SELECT_BY (TODO 2e)", SELECT_BY),
        ]
        if value is None
    ]

    if missing:
        raise SystemExit(
            "Fill these in before training: " + ", ".join(missing)
        )


def calculate_reference_distribution(values, bins=PSI_BINS):
    """
    Create the reference distribution used later by monitoring.py.

    Returns:
        edges: fixed bin boundaries
        proportions: fraction of training observations in each bin
    """

    values = np.asarray(values, dtype=float)

    # Quantile-based bins from the training distribution.
    edges = np.percentile(
        values,
        np.linspace(0, 100, bins + 1),
    )

    # Duplicate values can create duplicate edges.
    edges = np.unique(edges)

    if len(edges) < 2:
        raise ValueError(
            "Cannot create PSI bins because the training feature "
            "has insufficient variation."
        )

    counts, _ = np.histogram(
        values,
        bins=edges,
    )

    proportions = counts / len(values)

    return edges.tolist(), proportions.tolist()


def save_monitoring_reference(df, training_predictions):
    """
    Save the training/reference distributions required by monitoring.py.
    """

    reference = {
        "psi_bins": PSI_BINS,
        "features": {},
        "prediction_probability": {},
    }

    # Feature reference distributions
    for feature in FEATURES:

        edges, proportions = calculate_reference_distribution(
            df[feature].values,
            bins=PSI_BINS,
        )

        reference["features"][feature] = {
            "edges": edges,
            "proportions": proportions,
        }

    # Prediction probability reference distribution
    edges, proportions = calculate_reference_distribution(
        training_predictions,
        bins=PSI_BINS,
    )

    reference["prediction_probability"] = {
        "edges": edges,
        "proportions": proportions,
    }

    with open(MONITORING_REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2)

    print(
        f"Saved monitoring reference distributions to "
        f"{MONITORING_REFERENCE_PATH}"
    )


def main():

    check_todos()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = make_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df[TARGET],
        test_size=0.2,
        stratify=df[TARGET],
        random_state=42,
    )

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            ),
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = {}

    # We'll save the probabilities from the final/best model.
    best_training_probabilities = None

    for name, model in candidates.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)

            proba = model.predict_proba(X_test)[:, 1]

            pred = (
                proba >= THRESHOLD
            ).astype(int)

            metrics = {
                "roc_auc": roc_auc_score(
                    y_test,
                    proba,
                ),

                "pr_auc": average_precision_score(
                    y_test,
                    proba,
                ),

                "recall": recall_score(
                    y_test,
                    pred,
                ),

                "precision": precision_score(
                    y_test,
                    pred,
                    zero_division=0,
                ),
            }

            mlflow.log_params({
                "model_type": name,
                "threshold": THRESHOLD,
                "n_rows": len(df),
            })

            mlflow.log_metrics(metrics)

            info = mlflow.sklearn.log_model(
                model,
                name="model",
                input_example=X_test.head(3),
                **SAVE_OPTIONS,
            )

            results[name] = {
                "metrics": metrics,
                "uri": info.model_uri,
                "model": model,
            }

            print(
                f"{name:20s}",
                {
                    k: round(v, 3)
                    for k, v in metrics.items()
                },
            )

    # Select winner
    best = max(
        results,
        key=lambda n: results[n]["metrics"][SELECT_BY],
    )

    best_metrics = results[best]["metrics"]

    print(
        f"\nWinner by {SELECT_BY}: {best}"
    )

    if best_metrics["pr_auc"] < MIN_PR_AUC:

        raise SystemExit(
            f"Quality gate FAILED: "
            f"{best} has PR-AUC "
            f"{best_metrics['pr_auc']:.3f} "
            f"< {MIN_PR_AUC}"
        )

    # -----------------------------------------------------
    # Save monitoring reference distributions
    # -----------------------------------------------------

    best_model = results[best]["model"]

    # These are probabilities generated by the selected model.
    training_probabilities = best_model.predict_proba(
        df[FEATURES]
    )[:, 1]

    save_monitoring_reference(
        df,
        training_probabilities,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    joblib.dump(
        best_model,
        "model.joblib",
    )

    print("Saved model.joblib")

    # -----------------------------------------------------
    # Register model
    # -----------------------------------------------------

    version = mlflow.register_model(
        results[best]["uri"],
        MODEL_NAME,
    ).version

    if version is None:
        raise SystemExit(
            "Model registration failed"
        )

    MlflowClient().set_registered_model_alias(
        MODEL_NAME,
        "production",
        version,
    )

    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    with open("metrics.json", "w") as f:

        json.dump(
            {
                "best_model": best,
                "selected_by": SELECT_BY,
                "threshold": THRESHOLD,
                "min_pr_auc": MIN_PR_AUC,
                "registered_version": version,
                **best_metrics,
            },
            f,
            indent=2,
        )

    print(
        f"Registered {MODEL_NAME} "
        f"v{version} as @production"
    )


if __name__ == "__main__":
    main()