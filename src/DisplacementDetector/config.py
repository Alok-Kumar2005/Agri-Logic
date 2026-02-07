from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

EGMS_DATA_DIR = BASE_DIR / "data" / "raw" / "egms"

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MIN_COHERENCE = 0.6

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "n_jobs": -1,
    "random_state": 42,
}
