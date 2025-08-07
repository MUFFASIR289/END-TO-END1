import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

class DataIngestion:
    def __init__(self):
        self.train_data_path = os.path.join("artifacts", "train.csv")
        self.test_data_path = os.path.join("artifacts", "test.csv")
        self.raw_data_path = os.path.join("artifacts", "raw.csv")

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")

        try:
            df = pd.read_csv("notebook/data/study.csv")
            logging.info("Dataset loaded successfully")

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            df.to_csv(self.raw_data_path, index=False, header=True)

            logging.info("Train-test split started")
            from sklearn.model_selection import train_test_split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.train_data_path, index=False, header=True)
            test_set.to_csv(self.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            return self.train_data_path, self.test_data_path, self.raw_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data, raw_data = obj.initiate_data_ingestion()

    from src.components.data_transformation import DataTransformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_transformation(train_data, test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr,preprocessor_path=preprocessor_path))

    