"""FastAPI service that predicts machine failure from one sensor reading."""
import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import json
from pathlib import Path

from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

from database import init_db, insert_prediction, get_connection

FEATURES = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]

# TODO 3d: use the same THRESHOLD you chose in train.py (TODO 2a). Replace "____" with that number, e.g. "0.3".
# THINK: two copies of one number can drift apart. How could the API read it from MLflow or metrics.json?
metrics = json.loads(Path("metrics.json").read_text())
DEFAULT_THRESHOLD = metrics["threshold"]
if DEFAULT_THRESHOLD == "____":
    raise RuntimeError("TODO 3d in app.py: set DEFAULT_THRESHOLD to the threshold you chose in train.py")
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("machine-failure-api")

REQUESTS = Counter(
    "machine_failure_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)

LATENCY = Histogram(
    "machine_failure_api_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
)

PREDICTIONS = Counter(
    "machine_failure_predictions_total",
    "Total machine failure predictions",
    ["predicted"],
)

PREDICTION_PROBABILITY = Histogram(
    "machine_failure_prediction_probability",
    "Distribution of predicted machine failure probabilities",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5,
             0.6, 0.7, 0.8, 0.9, 1.0),
)

MODEL_THRESHOLD = Gauge(
    "machine_failure_model_threshold",
    "Failure probability threshold currently used by the API",
)

MODEL_THRESHOLD.set(THRESHOLD)


def load_model():
    uri = os.getenv("MODEL_URI")                 # e.g. models:/machine-failure-model@production
    if uri:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(uri), uri
    path = os.getenv("MODEL_PATH", "model.joblib")
    return joblib.load(path), path


model, MODEL_SOURCE = load_model()
log.info("Model loaded from %s", MODEL_SOURCE)

app = FastAPI(title="Machine Failure Prediction API", version="1.0.0",
              description="Predicts whether a milling machine will fail soon, from one sensor reading.")

init_db()

# Prometheus /metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# TODO 3a: set limits (ge = minimum, le = maximum) so impossible readings get a 422 error.
# One field is done for you as an example. Use df.describe() from Part 1:
#   air_temp_k            292.7 - 306.9    process_temp_k  301.8 - 317.9
#   rotational_speed_rpm  1170  - 2287     torque_nm       9.5   - 67.0     tool_wear_min  0 - 240
# Options for how wide to make the limits:
#   Option A: exactly the training min/max    -> safest for the model, but rejects some real readings
#   Option B: physical limits of the machine  -> e.g. torque 0-100 Nm, tool wear 0-300 min (needs domain input)
# THINK: what should happen to a reading that is possible but outside the training data? Reject? Warn?
class SensorReading(BaseModel):
    air_temp_k: float = Field(..., ge=290, le=310, description="Air temperature (K)", examples=[300.0])
    process_temp_k: float = Field(..., ge=300, le=320, description="Process temperature (K)", examples=[310.5])
    rotational_speed_rpm: float = Field(..., ge=1169, le=2290, description="Spindle speed", examples=[1550])
    torque_nm: float = Field(..., ge=9, le=70, description="Torque (Nm)", examples=[62.0])
    tool_wear_min: float = Field(..., ge=0, le=241, description="Tool wear (minutes)", examples=[230])


class Prediction(BaseModel):
    prediction_id: int
    failure_probability: float
    failure_predicted: bool
    recommended_action: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    # TODO 3c: return a dict with the keys "model_source", "threshold" and "features".
    # HINT: the values are MODEL_SOURCE, THRESHOLD and FEATURES (all defined above).
    # THINK: who would call this endpoint, and when? (an engineer debugging a strange prediction?)
    return {
        "model_source": MODEL_SOURCE,
        "threshold": THRESHOLD,
        "features": FEATURES
    }


@app.post("/predict", response_model=Prediction)
def predict(reading: SensorReading):
    start = time.perf_counter()
    try:
        X = pd.DataFrame([reading.model_dump()])[FEATURES]

        probability = float(
            model.predict_proba(X)[0, 1]
        )

        predicted = probability >= THRESHOLD

        action = (
            "Schedule maintenance this shift"
            if predicted
            else "No action needed"
        )

        # ------------------------------
        # Update Prometheus metrics
        # ------------------------------

        PREDICTIONS.labels(
            predicted=str(predicted).lower()
        ).inc()

        PREDICTION_PROBABILITY.observe(probability)

        REQUESTS.labels(
            endpoint="/predict",
            status="200",
        ).inc()

        prediction_id = insert_prediction(
            timestamp=time.time(),
            reading=reading,
            prediction_probability=probability,
            predicted_failure=predicted,
        )

        return Prediction(
            prediction_id=prediction_id,
            failure_probability=round(probability, 4),
            failure_predicted=predicted,
            recommended_action=action,
        )

    except Exception:

        REQUESTS.labels(
            endpoint="/predict",
            status="500",
        ).inc()

        raise

    finally:

        LATENCY.labels(
            endpoint="/predict"
        ).observe(
            time.perf_counter() - start
        )

@app.post("/outcome/{prediction_id}")
def record_outcome(
    prediction_id: int,
    actual_failure: bool,
):
    conn = get_connection()

    cursor = conn.execute(
        """
        UPDATE predictions
        SET actual_failure = ?
        WHERE id = ?
        """,
        (
            int(actual_failure),
            prediction_id,
        ),
    )

    conn.commit()
    updated = cursor.rowcount
    conn.close()

    if updated == 0:
        return {
            "error": "prediction not found",
            "prediction_id": prediction_id,
        }

    return {
        "prediction_id": prediction_id,
        "actual_failure": actual_failure,
    }