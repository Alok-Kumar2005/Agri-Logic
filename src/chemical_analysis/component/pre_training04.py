import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logging import logging
from src.exception import CustomException


class PreTraining:
    def __init__( self, X_path: str, y_path: str, params_path: str, models_dir: str, metrics_path: str ):
        self.X_path = X_path
        self.y_path = y_path
        self.params_path = params_path
        self.models_dir = models_dir
        self.metrics_path = metrics_path
        try:
            logging.info("Loading features and targets")
            self.X = np.load(self.X_path)
            self.y = np.load(self.y_path)
            self.target_cols = ["N", "P", "K", "pH"]
        except Exception as e:
            logging.error("Error loading data")
            raise CustomException(e, sys)
        try:
            logging.info("Loading model parameters from YAML")
            with open(self.params_path, "r") as f:
                self.params = yaml.safe_load(f)
        except Exception as e:
            logging.error("Error loading params.yaml")
            raise CustomException(e, sys)

        os.makedirs(self.models_dir, exist_ok=True)

        self.metrics = {}

    def train_and_save(self):
        try:
            for i, target in enumerate(self.target_cols):
                logging.info(f"Training model for target: {target}")

                y_target = self.y[:, i]

                param = self.params.get(target)
                if param is None:
                    raise ValueError(f"No params found for target {target}")

                model = XGBRegressor(
                    n_estimators=param.get("n_estimators", 100),
                    max_depth=param.get("max_depth", 6),
                    learning_rate=param.get("lr", 0.1),
                    random_state=42,
                    objective="reg:squarederror"
                )
                model.fit(self.X, y_target)
                y_pred = model.predict(self.X)
                mse = mean_squared_error(y_target, y_pred)
                r2 = r2_score(y_target, y_pred)
                mae = mean_absolute_error(y_target, y_pred)

                self.metrics[target] = {"mse": mse, "r2": r2, "mae": mae}
                model_file = os.path.join(self.models_dir, f"xgb_{target}.joblib")
                joblib.dump(model, model_file)
                logging.info(f"Saved model for {target} at {model_file}")

            with open(self.metrics_path, "w") as f:
                for t, m in self.metrics.items():
                    f.write(f"{t} metrics:\n")
                    f.write(f"RMSE: {m['mse']:.4f}\n")
                    f.write(f"R2: {m['r2']:.4f}\n")
                    f.write(f"MAE: {m['mae']:.4f}\n\n")

            logging.info(f"Metrics saved at {self.metrics_path}")

        except Exception as e:
            logging.error("Error during pretraining")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        pretrainer = PreTraining(
            X_path="artifacts/X_lucas.npy",
            y_path="artifacts/y_lucas.npy",
            params_path="params.yaml",
            models_dir="artifacts/models",
            metrics_path="artifacts/metrics.txt"
        )
        pretrainer.train_and_save()
    except Exception as e:
        raise CustomException(e, sys)
