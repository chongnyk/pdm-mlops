import requests
import random

URL = "http://localhost:8000/predict"

for i in range(100):
    reading = {
        "air_temp_k": random.uniform(295, 305),
        "process_temp_k": random.uniform(305, 315),
        "rotational_speed_rpm": random.uniform(1300, 2200),
        "torque_nm": random.uniform(15, 65),
        "tool_wear_min": random.uniform(0, 240),
    }

    response = requests.post(URL, json=reading)

    if response.ok:
        result = response.json()
        print(
            i + 1,
            "prediction_id =", result["prediction_id"],
            "probability =", result["failure_probability"],
        )
    else:
        print("ERROR:", response.status_code, response.text)