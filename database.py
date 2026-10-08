import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "data/machine_failure.db")


def get_connection():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,

            air_temp_k REAL NOT NULL,
            process_temp_k REAL NOT NULL,
            rotational_speed_rpm REAL NOT NULL,
            torque_nm REAL NOT NULL,
            tool_wear_min REAL NOT NULL,

            prediction_probability REAL NOT NULL,
            predicted_failure INTEGER NOT NULL,

            actual_failure INTEGER
        )
    """)

    conn.commit()
    conn.close()


def insert_prediction(
    timestamp,
    reading,
    prediction_probability,
    predicted_failure,
    actual_failure=None,
):
    conn = get_connection()

    cursor = conn.execute(
        """
        INSERT INTO predictions (
            timestamp,
            air_temp_k,
            process_temp_k,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_min,
            prediction_probability,
            predicted_failure,
            actual_failure
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            reading.air_temp_k,
            reading.process_temp_k,
            reading.rotational_speed_rpm,
            reading.torque_nm,
            reading.tool_wear_min,
            prediction_probability,
            int(predicted_failure),
            actual_failure,
        ),
    )

    conn.commit()

    prediction_id = cursor.lastrowid

    conn.close()

    return prediction_id