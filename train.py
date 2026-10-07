"""Train candidate models, track them in MLflow, register the best one as @production."""
import inspect
import json
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# Extra models you may use in TODO 2c:
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from data import FEATURES, TARGET, make_data

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "machine-failure"
MODEL_NAME = "machine-failure-model"

# TODO 2a: decision THRESHOLD. If the predicted failure probability >= THRESHOLD, we raise an alarm.
#   Option A: 0.5  -> the usual default, fewer false alarms, more missed failures
#   Option B: 0.3  -> catches more failures (higher recall), more false alarms
#   Option C: 0.2  -> very cautious
# THINK: a missed failure stops the line for hours; a false alarm costs one 10-minute inspection.
THRESHOLD = 0.3

# TODO 2b: quality gate. The best model is NOT registered if its PR-AUC is below this value.
#   Option A: 0.5 (lenient)   Option B: 0.6   Option C: 0.7 (strict)
# HINT: the lazy model from Part 1 would score about the failure rate (~0.06).
MIN_PR_AUC = 0.7

# TODO 2e: which metric picks the winner? (used further down)
#   Option A: "pr_auc"  -> ranking quality on rare failures, does not depend on THRESHOLD
#   Option B: "recall"  -> share of failures caught at your THRESHOLD
#   Option C: "roc_auc" -> overall separation, can look good even on very rare classes
# THINK: try "recall" once. Which model wins, and does it pass your quality gate?
SELECT_BY = "roc_auc"

# Newer MLflow saves sklearn models with skops (safer than pickle). skops only loads object types it has
# been told to trust. Tree-based models contain "sklearn.tree._tree.Tree" objects, which are safe here
# because this script created them. If you add a model in TODO 2c and get an error saying
# "Untrusted types found in the file: [...]", add the type named in that error to this list.
# THINK: why would you NOT want to trust every type automatically? (security slide!)
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]
SAVE_OPTIONS = {}
if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
    SAVE_OPTIONS["skops_trusted_types"] = TRUSTED_TYPES


def check_todos():
    missing = [name for name, value in [("THRESHOLD (TODO 2a)", THRESHOLD), ("MIN_PR_AUC (TODO 2b)", MIN_PR_AUC),
                                        ("SELECT_BY (TODO 2e)", SELECT_BY)] if value is None]
    if missing:
        raise SystemExit("Fill these in before training: " + ", ".join(missing))


def main():
    check_todos()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.2, stratify=df[TARGET], random_state=42)

    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42),
        # TODO 2c (recommended): add a third candidate model. Pick ONE and uncomment it by deleting the
        #   "#   Option X: " part at the start of its line. Then compare all models in MLflow.
        "decision_tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        #   Option B: "extra_trees": ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        #   Option C: "gradient_boosting": GradientBoostingClassifier(random_state=42),
        #   Option D: "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        # THINK: a single decision tree is easy to explain to engineers. Is it worth the lower score?
    }

    results = {}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]          # probability of failure, 0..1
            pred = (proba >= THRESHOLD).astype(int)           # 1 = raise an alarm

            # TODO 2d: compute recall and precision from y_test and pred.
            # HINT: recall_score(y_test, pred)   and   precision_score(y_test, pred, zero_division=0)
            # THINK: in plain words, what does each one mean for the maintenance team?
            metrics = {
                "roc_auc": roc_auc_score(y_test, proba),
                "pr_auc": average_precision_score(y_test, proba),
                "recall": recall_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
            }
            if metrics["recall"] is None or metrics["precision"] is None:
                raise SystemExit("TODO 2d: fill in recall and precision")
            mlflow.log_params({"model_type": name, "threshold": THRESHOLD, "n_rows": len(df)})
            mlflow.log_metrics(metrics)
            info = mlflow.sklearn.log_model(model, name="model", input_example=X_test.head(3), **SAVE_OPTIONS)
            results[name] = {"metrics": metrics, "uri": info.model_uri, "model": model}
            print(f"{name:20s}", {k: round(v, 3) for k, v in metrics.items()})

    best = max(results, key=lambda n: results[n]["metrics"][SELECT_BY])
    best_metrics = results[best]["metrics"]
    print(f"\nWinner by {SELECT_BY}: {best}")
    if best_metrics["pr_auc"] < MIN_PR_AUC:
        raise SystemExit(f"Quality gate FAILED: {best} has PR-AUC {best_metrics['pr_auc']:.3f} < {MIN_PR_AUC}")

    # Save the model file first: the API tests and the Docker image use it
    joblib.dump(results[best]["model"], "model.joblib")
    print("Saved model.joblib")

    # TODO 2f: register the winner in the MLflow Model Registry and point the alias "production" at it.
    # HINT:  version = mlflow.register_model(results[best]["uri"], MODEL_NAME).version
    #        MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)
    # THINK: the API will load "models:/machine-failure-model@production". Why is an alias better than
    #        writing version 3 into the API code? How would you roll back?
    version = mlflow.register_model(results[best]["uri"], MODEL_NAME).version
    if version is None:
        raise SystemExit("TODO 2f: register the model and set the production alias")

    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version)

    with open("metrics.json", "w") as f:
        json.dump({"best_model": best, "selected_by": SELECT_BY, "threshold": THRESHOLD,
                   "min_pr_auc": MIN_PR_AUC, "registered_version": version, **best_metrics}, f, indent=2)


    print(f"Registered {MODEL_NAME} v{version} as @production")


if __name__ == "__main__":
    main()
