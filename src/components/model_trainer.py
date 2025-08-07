import pandas as pd
import numpy as np
import os
import sys
from dataclasses import dataclass

from src.utils import evaluate_model, save_object
from src.exception import CustomException
from src.logger import logging

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_Train, Y_Train, X_Test, Y_Test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor()
            }

            logging.info("Evaluating models")
            model_report = evaluate_model(
                x_train=X_Train, y_train=Y_Train,
                x_test=X_Test, y_test=Y_Test,
                models=models
            )

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception("No suitable model found with acceptable performance")

            logging.info(f"Best model selected: {best_model_name} with score: {best_model_score:.4f}")

            best_model.fit(X_Train, Y_Train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            pred = best_model.predict(X_Test)
            r2 = r2_score(Y_Test, pred)
            print(f"✅ Best model: {best_model_name} with R² score: {r2:.4f}")
            return r2

        except Exception as e:
            raise CustomException(e, sys)