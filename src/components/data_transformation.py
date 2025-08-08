import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['writing score', 'math score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_transformation(self, train_path, test_path):
        try:
            print("🚀 Data Transformation started")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print("✅ Train shape:", train_df.shape)
            print("✅ Test shape:", test_df.shape)

            preprocessor = self.get_data_transformer_obj()

            train_arr = preprocessor.fit_transform(train_df)
            test_arr = preprocessor.transform(test_df)

            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file), exist_ok=True)
            save_object(self.data_transformation_config.preprocessor_obj_file, preprocessor)

            print("✅ Preprocessor saved at:", self.data_transformation_config.preprocessor_obj_file)

            return train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file

        except Exception as e:
            raise CustomException(e, sys)