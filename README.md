# Predictive Maintenance MLOps

Predicts whether a CNC milling machine will fail soon from 5 sensor readings, and serves the model as an API.

| File | Purpose |
|---|---|
| `data.py` | Generates the sensor dataset (modelled on AI4I 2020) |
| `train.py` | Trains candidate models, tracks them in MLflow, registers the best as `@production` |
| `app.py` | FastAPI service: `/predict`, `/health`, `/model-info`, docs at `/docs` |
| `tests/` | Data and API tests (pytest) |
| `Dockerfile` | Container for the API |
| `.github/workflows/ci.yml` | Train, test, build and smoke-test on every push |

```bash
pip install -r requirements.txt
python train.py && python -m pytest -q
uvicorn app:app --reload                          # http://127.0.0.1:8000/docs
docker build -t machine-failure-api . && docker run -p 8000:8000 machine-failure-api
```
