import os
import sys
import gdown
import pandas as pd
from src.logging import logging
from src.exception import CustomException


class DataIngestion:
    def __init__(self, file_id: str , raw_data_path: str):
        self.file_id = file_id
        self.raw_data_path = raw_data_path

    def download_data(self) -> pd.DataFrame:
        try:
            logging.info("Starting data download from Google Drive")

            os.makedirs(os.path.dirname(self.raw_data_path), exist_ok=True)

            url = f"https://drive.google.com/uc?id={self.file_id}"
            gdown.download(url, self.raw_data_path, quiet=False)

            df = pd.read_csv(self.raw_data_path)

            logging.info("Data download completed successfully")
            return df

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
        
    def save_data(self, df: pd.DataFrame, file_path: str) -> None:
        try:
            logging.info(f"Saving data to {file_path}")

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            df.to_csv(file_path, index=False)

            logging.info("Data saved successfully")

        except Exception as e:
            logging.error("Error occurred while saving data")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    ## Loading lucas Data
    data_ingestion = DataIngestion("1dARVPvodCLUpK0SneyvDgLtTtcPwHiWW", "data/raw/lucas_training_data.csv")
    df = data_ingestion.download_data()
    data_ingestion.save_data(df, "data/raw/lucas_training_data.csv")

    ## loading Punjab data
    data_ingestion = DataIngestion("1nDPRHXi2iNPV8QSJRSm7caGN7itasFL5", "data/raw/punjab_soil_samples.csv")
    df = data_ingestion.download_data()
    data_ingestion.save_data(df, "data/raw/punjab_soil_samples.csv")