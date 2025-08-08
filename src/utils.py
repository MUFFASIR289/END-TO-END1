import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"📦 Object saved to: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    report = {}
    for name, model in models.items():
        model_params = param.get(name, {})
        grid = GridSearchCV(model, model_params, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(x_train, y_train)
        best_model = grid.best_estimator_
        y_pred = best_model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ {name} — R²: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        report[name] = r2
    return report