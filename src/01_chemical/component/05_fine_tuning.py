import os
import sys
import yaml
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.logging import logging
from src.exception import CustomException


class FineTuning:
    def __init__( self, X_path: str, y_path: str, params_path: str, pretrained_models_dir: str, finetuned_models_dir: str, metrics_path: str ):
        try:
            logging.info("Loading Punjab fine-tuning data")
            self.X = np.load(X_path)
            self.y = np.load(y_path)
            self.target_cols = ["N", "P", "K", "pH"]

            with open(params_path, "r") as f:
                self.params = yaml.safe_load(f)

            self.pretrained_models_dir = pretrained_models_dir
            self.finetuned_models_dir = finetuned_models_dir
            self.metrics_path = metrics_path
            os.makedirs(self.finetuned_models_dir, exist_ok=True)
            self.metrics = {}

        except Exception as e:
            logging.error("Error initializing FineTuning")
            raise CustomException(e, sys)

    def finetune_and_save(self):
        try:
            for i, target in enumerate(self.target_cols):
                logging.info(f"Fine-tuning model for target: {target}")

                pretrained_path = os.path.join(
                    self.pretrained_models_dir, f"xgb_{target}.joblib"
                )

                if not os.path.exists(pretrained_path):
                    raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

                model: XGBRegressor = joblib.load(pretrained_path)
                ft_params = self.params.get(f"{target}_FT")
                if ft_params is None:
                    raise ValueError(f"Missing fine-tuning params for {target}_FT")

                model.set_params(
                    n_estimators=ft_params["n_estimators"],
                    max_depth=ft_params["max_depth"],
                    learning_rate=ft_params["lr"]
                )

                y_target = self.y[:, i]
                model.fit(self.X, y_target, xgb_model=model.get_booster() )
                y_pred = model.predict(self.X)
                mse = mean_squared_error(y_target, y_pred)
                r2 = r2_score(y_target, y_pred)
                mae = mean_absolute_error(y_target, y_pred)

                self.metrics[target] = {
                    "mse": mse,
                    "r2": r2,
                    "mae": mae
                }

                # Save fine-tuned model
                finetuned_path = os.path.join(
                    self.finetuned_models_dir, f"xgb_{target}_finetuned.joblib"
                )
                joblib.dump(model, finetuned_path)
                logging.info(f"Saved fine-tuned model: {finetuned_path}")

            with open(self.metrics_path, "w") as f:
                for t, m in self.metrics.items():
                    f.write(f"{t} Fine-Tuning Metrics:\n")
                    f.write(f"MSE: {m['mse']:.4f}\n")
                    f.write(f"R2: {m['r2']:.4f}\n")
                    f.write(f"MAE: {m['mae']:.4f}\n\n")

            logging.info(f"Fine-tuning metrics saved at {self.metrics_path}")

        except Exception as e:
            logging.error("Error during fine-tuning")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        finetuner = FineTuning(
            X_path="artifacts/X_punjab.npy",
            y_path="artifacts/y_punjab.npy",
            params_path="params.yaml",
            pretrained_models_dir="artifacts/models",
            finetuned_models_dir="artifacts/finetuned_models",
            metrics_path="artifacts/finetune_metrics.txt"
        )

        finetuner.finetune_and_save()

    except Exception as e:
        raise CustomException(e, sys)
