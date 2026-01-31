import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.logging import logging
from src.exception import CustomException


class InferencePipeline:
    def __init__(self, use_finetuned: bool = True):
        self.use_finetuned = use_finetuned
        self.target_cols = ["N", "P", "K", "pH"]
        self.feature_cols = [
            'B11', 'B12', 'B2', 'B3', 'B4', 'B8', 'Evap_tavg', 'NDVI', 'NDWI', 'Rainf_tavg', 'SAVI',
            'SoilMoi0_10cm_inst', 'Tair_f_inst', 'elevation', 'slope'
        ]
        self.preprocessor = None
        self.models = {}
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            preprocessor_path = "artifacts/preprocessor.joblib"
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}")
            
            self.preprocessor = joblib.load(preprocessor_path)

            if self.use_finetuned:
                models_dir = "artifacts/finetuned_models"
                model_suffix = "_finetuned"
            else:
                models_dir = "artifacts/models"
                model_suffix = ""

            for target in self.target_cols:
                model_path = os.path.join(models_dir, f"xgb_{target}{model_suffix}.joblib")
                
                if not os.path.exists(model_path):
                    if self.use_finetuned:
                        logging.warning(f"Finetuned model not found for {target}, using pretrained")
                        model_path = os.path.join("artifacts/models", f"xgb_{target}.joblib")
                    
                    if not os.path.exists(model_path):
                        raise FileNotFoundError(f"Model not found: {model_path}")
                
                self.models[target] = joblib.load(model_path)
                logging.info(f"Loaded model for {target} from {model_path}")
        except Exception as e:
            logging.error("Error loading inference artifacts")
            raise CustomException(e, sys)
    
    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            missing_cols= set(self.feature_cols) -set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing features: {missing_cols}")
            
            df_features = df[self.feature_cols].copy()
            if df_features.isnull().any().any():
                logging.warning("NaN values detected in features. Filling with median.")
                df_features = df_features.fillna(df_features.median())
            return df_features
        except Exception as e:
            raise CustomException(e, sys)
        
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        try:
            df = pd.DataFrame([features])
            df = self._validate_features(df)
            X_transformed = self.preprocessor.transform(df)

            predictions = {}
            for target in self.target_cols:
                pred = self.models[target].predict(X_transformed)[0]
                predictions[target] = float(pred)
            return predictions
        except Exception as e:
            logging.error("Error during single prediction")
            raise CustomException(e, sys)
        
    def predict_batch(self, df: pd.DataFrame)-> pd.DataFrame:
        try:
            logging.info(f"Batch prediction for {len(df)} samples")
            df_features = self._validate_features(df)
            X_transformed = self.preprocessor.transform(df_features)
            predictions_dict = {}
            for target in self.target_cols:
                predictions = self.models[target].predict(X_transformed)
                predictions_dict[f"{target}_predicted"] = predictions
            results = df.copy()
            for target, preds in predictions_dict.items():
                results[target] = preds

            return results
        except Exception as e:
            logging.error("Error in batch prediction")
            raise CustomException(e, sys)

    def predict_from_earth_engine_data(self, ee_data: pd.DataFrame) -> pd.DataFrame:
        try:
            results = self.predict_batch(ee_data)
            results = self._add_recommendations(results)
            return results
        except Exception as e:
            raise CustomException(e, sys)
    
    def _add_recommendations(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            recommendations = []
            
            for _, row in df.iterrows():
                n_val = row.get('N_predicted', 0)
                p_val = row.get('P_predicted', 0)
                k_val = row.get('K_predicted', 0)
                ph_val = row.get('pH_predicted', 7.0)
                
                rec_parts = []
                
                # Nitrogen recommendation (ideal: 250-350 mg/kg)
                if n_val < 200:
                    rec_parts.append(f"Apply {int((250 - n_val) * 0.15)}kg/ha Nitrogen")
                elif n_val > 400:
                    rec_parts.append("Reduce Nitrogen application")
                
                # Phosphorus recommendation (ideal: 30-60 mg/kg)
                if p_val < 25:
                    rec_parts.append(f"Apply {int((40 - p_val) * 0.2)}kg/ha Phosphorus")
                elif p_val > 80:
                    rec_parts.append("Reduce Phosphorus application")
                
                # Potassium recommendation (ideal: 150-250 mg/kg)
                if k_val < 120:
                    rec_parts.append(f"Apply {int((180 - k_val) * 0.12)}kg/ha Potassium")
                elif k_val > 300:
                    rec_parts.append("Reduce Potassium application")
                
                # pH recommendation (ideal: 6.0-7.5)
                if ph_val < 5.5:
                    rec_parts.append("Apply lime to increase pH")
                elif ph_val > 8.0:
                    rec_parts.append("Apply sulfur to decrease pH")
                
                if not rec_parts:
                    recommendations.append("Soil nutrients are balanced - maintain current practices")
                else:
                    recommendations.append("; ".join(rec_parts))
            
            df['recommendation'] = recommendations
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model_info(self) -> Dict:
        return {
            "using_finetuned": self.use_finetuned,
            "models_loaded": list(self.models.keys()),
            "feature_count": len(self.feature_cols),
            "features": self.feature_cols
        }
    
if __name__ == "__main__":
    pipeline = InferencePipeline(use_finetuned= True)
    sample_features = {
            'B11': 0.2345,
            'B12': 0.1876,
            'B2': 0.0543,
            'B3': 0.0678,
            'B4': 0.0821,
            'B8': 0.3456,
            'Evap_tavg': 0.000012,
            'NDVI': 0.65,
            'NDWI': 0.25,
            'Rainf_tavg': 0.000015,
            'SAVI': 0.25,
            'SoilMoi0_10cm_inst': 0.28,
            'Tair_f_inst': 298.5,
            'elevation': 245.0,
            'slope': 3.2
        }
    predictions = pipeline.predict(sample_features)
    for nutrient, value in predictions.items():
        print(f"  {nutrient}: {value:.2f}")