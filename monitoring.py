import json
import os
import sqlite3

import numpy as np

from fastapi import FastAPI
from prometheus_client import Gauge, make_asgi_app


DB_PATH = os.getenv(
    "DB_PATH",
    "/app/data/machine_failure.db",
)

REFERENCE_PATH = os.getenv(
    "REFERENCE_PATH",
    "/app/monitoring_reference.json",
)

ROLLING_WINDOW = int(
    os.getenv("ROLLING_WINDOW", "100")
)


FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


# ---------------------------------------------------------
# Load training/reference distributions
# ---------------------------------------------------------

with open(REFERENCE_PATH) as f:
    REFERENCE = json.load(f)


# ---------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------

FEATURE_PSI = Gauge(
    "machine_failure_feature_psi",
    "PSI between live feature distribution and training distribution",
    ["feature"],
)

PREDICTION_PSI = Gauge(
    "machine_failure_prediction_psi",
    "PSI between live prediction probabilities and training probabilities",
)

ROLLING_RECALL = Gauge(
    "machine_failure_rolling_recall",
    "Recall over recent predictions with known outcomes",
)

KNOWN_OUTCOMES = Gauge(
    "machine_failure_known_outcomes",
    "Number of predictions with known actual outcomes",
)


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def get_connection():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# ---------------------------------------------------------
# PSI
# ---------------------------------------------------------

def calculate_psi(
    reference_proportions,
    current_values,
    edges,
):
    """
    Calculate PSI using fixed training bins.

    reference_proportions:
        Proportion of training observations in each bin.

    current_values:
        Recent/live observations.

    edges:
        Bin boundaries created during training.
    """

    if len(current_values) == 0:
        return None

    reference_pct = np.asarray(
        reference_proportions,
        dtype=float,
    )

    current_counts, _ = np.histogram(
        current_values,
        bins=np.asarray(edges),
    )

    current_pct = (
        current_counts / len(current_values)
    )

    # Avoid division by zero / log(0).
    reference_pct = np.clip(
        reference_pct,
        1e-6,
        None,
    )

    current_pct = np.clip(
        current_pct,
        1e-6,
        None,
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(
            current_pct / reference_pct
        )
    )

    return float(psi)


# ---------------------------------------------------------
# Database queries
# ---------------------------------------------------------

def get_recent_predictions(
    limit=ROLLING_WINDOW,
):

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM predictions
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

    conn.close()

    return list(reversed(rows))


# ---------------------------------------------------------
# Rolling recall
# ---------------------------------------------------------

def calculate_recall(rows):

    rows = [
        row
        for row in rows
        if row["actual_failure"] is not None
    ]

    if not rows:
        return None

    tp = sum(
        row["predicted_failure"] == 1
        and row["actual_failure"] == 1
        for row in rows
    )

    fn = sum(
        row["predicted_failure"] == 0
        and row["actual_failure"] == 1
        for row in rows
    )

    if tp + fn == 0:
        return None

    return tp / (tp + fn)


# ---------------------------------------------------------
# Monitoring update
# ---------------------------------------------------------

def update_metrics():

    rows = get_recent_predictions()

    if not rows:
        return

    # -----------------------------------------------------
    # Feature PSI
    # -----------------------------------------------------

    for feature in FEATURES:

        current = np.array(
            [
                row[feature]
                for row in rows
            ],
            dtype=float,
        )

        reference = REFERENCE["features"][feature]

        psi = calculate_psi(
            reference_proportions=reference["proportions"],
            current_values=current,
            edges=reference["edges"],
        )

        if psi is not None:

            FEATURE_PSI.labels(
                feature=feature
            ).set(psi)

    # -----------------------------------------------------
    # Prediction PSI
    # -----------------------------------------------------

    current_predictions = np.array(
        [
            row["prediction_probability"]
            for row in rows
        ],
        dtype=float,
    )

    reference_predictions = REFERENCE[
        "prediction_probability"
    ]

    psi = calculate_psi(
        reference_proportions=reference_predictions[
            "proportions"
        ],
        current_values=current_predictions,
        edges=reference_predictions[
            "edges"
        ],
    )

    if psi is not None:
        PREDICTION_PSI.set(psi)

    # -----------------------------------------------------
    # Rolling recall
    # -----------------------------------------------------

    recall = calculate_recall(rows)

    if recall is not None:
        ROLLING_RECALL.set(recall)

    # -----------------------------------------------------
    # Number of predictions with known outcomes
    # -----------------------------------------------------

    KNOWN_OUTCOMES.set(
        sum(
            row["actual_failure"] is not None
            for row in rows
        )
    )


# ---------------------------------------------------------
# FastAPI monitoring service
# ---------------------------------------------------------

app = FastAPI(
    title="Machine Failure Monitoring",
)


@app.get("/update")
def update():

    update_metrics()

    return {
        "status": "updated",
    }


# ---------------------------------------------------------
# Prometheus endpoint
# ---------------------------------------------------------

metrics_app = make_asgi_app()

app.mount(
    "/metrics",
    metrics_app,
)