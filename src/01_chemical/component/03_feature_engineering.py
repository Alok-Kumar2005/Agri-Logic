import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, PowerTransformer, QuantileTransformer
)
from src.logging import logging
from src.exception import CustomException


class FeatureEngineering:
    def __init__( self, preprocessor_path: str, X_save_path: str, y_save_path: str ):
        self.preprocessor_path = preprocessor_path
        self.X_save_path = X_save_path
        self.y_save_path = y_save_path
        self.preprocessor = None
        self.target_cols = ["N", "P", "K", "pH"]
        self.normal_cols = [ "B11", "B12", "B8", "Evap_tavg", "SoilMoi0_10cm_inst", "Tair_f_inst" ]
        self.log_skewed_cols = [ "B2", "B3", "B4", "elevation", "slope" ]
        self.bounded_cols = ["NDVI", "NDWI", "SAVI"]
        self.rain_cols = ["Rainf_tavg"]

    def build_preprocessor(self):
        try:
            logging.info("Building feature engineering preprocessor")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("normal", StandardScaler(), self.normal_cols),
                    ("skewed", PowerTransformer(method="yeo-johnson"), self.log_skewed_cols),
                    ("bounded", PowerTransformer(method="yeo-johnson"), self.bounded_cols),
                    ("rain", QuantileTransformer(
                        n_quantiles=100,
                        output_distribution="normal"
                    ), self.rain_cols),
                ],
                remainder="drop"
            )
            return self.preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def _split_xy(self, df: pd.DataFrame):
        try:
            X = df.drop(columns=self.target_cols)
            y = df[self.target_cols]
            return X, y
        except Exception as e:
            raise CustomException(e, sys)

    def fit_transform(self, df: pd.DataFrame):
        try:
            logging.info("Splitting X and y")
            X_df, y_df = self._split_xy(df)

            logging.info("Fitting and transforming X features")

            if self.preprocessor is None:
                self.build_preprocessor()

            X_transformed = self.preprocessor.fit_transform(X_df)

            self._save_outputs(X_transformed, y_df)
            self._save_preprocessor()

            return X_transformed, y_df.values

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, df: pd.DataFrame):
        try:
            logging.info("Splitting X and y")
            X_df, y_df = self._split_xy(df)

            logging.info("Transforming X using saved preprocessor")

            if self.preprocessor is None:
                self.preprocessor = joblib.load(self.preprocessor_path)

            X_transformed = self.preprocessor.transform(X_df)

            self._save_outputs(X_transformed, y_df)

            return X_transformed, y_df.values

        except Exception as e:
            raise CustomException(e, sys)

    def _save_preprocessor(self):
        try:
            logging.info("Saving feature engineering preprocessor")
            os.makedirs(os.path.dirname(self.preprocessor_path), exist_ok=True)
            joblib.dump(self.preprocessor, self.preprocessor_path)
        except Exception as e:
            raise CustomException(e, sys)

    def _save_outputs(self, X: np.ndarray, y: pd.DataFrame):
        try:
            logging.info("Saving transformed X and raw y")

            os.makedirs(os.path.dirname(self.X_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.y_save_path), exist_ok=True)

            np.save(self.X_save_path, X)
            np.save(self.y_save_path, y.values)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Running Feature Engineering for LUCAS dataset")
        lucas_df = pd.read_csv("data/processed/lucas_training_data.csv")

        fe_lucas = FeatureEngineering(
            preprocessor_path="artifacts/preprocessor.joblib",
            X_save_path="artifacts/X_lucas.npy",
            y_save_path="artifacts/y_lucas.npy"
        )

        X_lucas, y_lucas = fe_lucas.fit_transform(lucas_df)

        logging.info(
            f"LUCAS Feature Engineering completed | "
            f"X shape: {X_lucas.shape}, y shape: {y_lucas.shape}"
        )


        logging.info("Running Feature Engineering for Punjab dataset")

        punjab_df = pd.read_csv("data/processed/punjab_soil_samples.csv")

        fe_punjab = FeatureEngineering(
            preprocessor_path="artifacts/preprocessor.joblib",
            X_save_path="artifacts/X_punjab.npy",
            y_save_path="artifacts/y_punjab.npy"
        )

        X_punjab, y_punjab = fe_punjab.transform(punjab_df)

        logging.info(
            f"Punjab Feature Engineering completed | "
            f"X shape: {X_punjab.shape}, y shape: {y_punjab.shape}"
        )

    except Exception as e:
        raise CustomException(e, sys)
