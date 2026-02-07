import numpy as np
import pandas as pd
import pickle
import xgboost as xgb
import json  
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from src.DisplacementDetector import config
from src.DisplacementDetector.data_processor import DataProcessor


class DisplacementPredictor:
    def __init__(self):
        self.model = None
        self.processor = DataProcessor()

    def create_features(self, ts: pd.DataFrame) -> pd.DataFrame:
        df = ts.copy()

        df["days"] = (df["date"] - df["date"].min()).dt.days
        df["month"] = df["date"].dt.month
        df["doy"] = df["date"].dt.dayofyear

        df["lag1"] = df["displacement"].shift(1)
        df["lag2"] = df["displacement"].shift(2)
        df["lag3"] = df["displacement"].shift(3)

        df["roll_mean_3"] = df["displacement"].rolling(3).mean()
        df["trend"] = df["displacement"].diff()

        return df.dropna()

    def train(self, max_points: int = 5000):
        X_all, y_all = [], []

        count = 0
        for csv_file in config.EGMS_DATA_DIR.glob("*.csv"):
            df = pd.read_csv(csv_file)

            for _, row in df.iterrows():
                if row["temporal_coherence"] < config.MIN_COHERENCE:
                    continue

                ts = self.processor.extract_time_series(row.to_dict())
                if len(ts) < 20:
                    continue

                feats = self.create_features(ts)
                if len(feats) < 5:
                    continue

                X_all.append(feats.drop(columns=["date", "displacement"]).values)
                y_all.append(feats["displacement"].values)

                count += 1
                if count >= max_points:
                    break
            if count >= max_points:
                break

        X = np.vstack(X_all)
        y = np.concatenate(y_all)

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = xgb.XGBRegressor(**config.XGBOOST_PARAMS)
        self.model.fit(Xtr, ytr)

        preds = self.model.predict(Xte)
        
        mae_score = mean_absolute_error(yte, preds)
        r2_score_val = r2_score(yte, preds)

        print("MAE:", mae_score)
        print("R²:", r2_score_val)

        metrics = {
            "MAE": float(mae_score),
            "R2": float(r2_score_val)
        }
        result_path = Path("artifacts/result/forecastmodel.json")
        result_path.parent.mkdir(parents=True, exist_ok=True)

        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved → {result_path}")
        # ---------------------------------

        model_path = config.MODELS_DIR / "egms_xgb.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        print(f"Model saved → {model_path}")

    def load_model(self):
        model_path = config.MODELS_DIR / "egms_xgb.pkl"
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)


if __name__ == "__main__":
    import sys

    predictor = DisplacementPredictor()

    if "--train" in sys.argv:
        predictor.train()
    else:
        print("Run with:")
        print("python -m src.DisplacementDetector.ml_predictor --train")