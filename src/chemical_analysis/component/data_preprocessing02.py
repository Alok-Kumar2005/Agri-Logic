import os
import sys
import pandas as pd
from src.logging import logging
from src.exception import CustomException


class DataPreprocessing:
    def __init__(self, csv_path: str, output_path: str):
        try:
            logging.info("Reading CSV file")
            self.df = pd.read_csv(csv_path)
            self.output_path = output_path
        except Exception as e:
            logging.error("Error reading CSV file")
            raise CustomException(e, sys)

    def basic_preprocessing(self):
        try:
            logging.info("Basic Preprocessing .....")

            df = self.df.copy()
            df = df.dropna()
            df = df.drop_duplicates()

            df = df[
                [
                    'B11', 'B12', 'B2', 'B3', 'B4', 'B8',
                    'Evap_tavg', 'NDVI', 'NDWI',
                    'Rainf_tavg', 'SAVI',
                    'SoilMoi0_10cm_inst', 'Tair_f_inst',
                    'elevation', 'slope',
                    'N', 'K', 'P', 'pH'
                ]
            ]

            self.df = df
            return self.df
        except Exception as e:
            logging.error("Error in Basic Preprocessing")
            raise CustomException(e, sys)

    def outliers_handling(self):
        try:
            logging.info("Handling Outliers .......")

            df = self.df.copy()
            feature_cols = df.columns.difference(['N', 'K', 'P', 'pH'])

            for col in feature_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5*IQR
                upper_bound = Q3 + 1.5*IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            self.df = df
            return self.df

        except Exception as e:
            logging.error("Error in Outlier Handling")
            raise CustomException(e, sys)

    def save(self):
        try:
            logging.info("Saving processed data")

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)

            logging.info(f"Processed data saved at: {self.output_path}")

        except Exception as e:
            logging.error("Error while saving processed data")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting Data Preprocessing Pipeline")

            self.basic_preprocessing()
            self.outliers_handling()
            self.save()

            logging.info("Data Preprocessing Completed")
            return self.df
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    preprocessor1 = DataPreprocessing(csv_path= "data/raw/lucas_training_data.csv",
                                      output_path= "data/processed/lucas_training_data.csv")
    preprocessor1.run()

    preprocessor2 = DataPreprocessing(csv_path= "data/raw/punjab_soil_samples.csv",
                                      output_path= "data/processed/punjab_soil_samples.csv")
    preprocessor2.run()