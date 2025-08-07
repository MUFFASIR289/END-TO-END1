import os
import pickle
import sys
import numpy as np
import pandas as pd
import dill

from src.exception import CustomException



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        print(f"📦 Object saved to: {file_path}")

    except Exception as e:
        raise e
    



from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(x_train, y_train, x_test, y_test, models):
    report = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        report[name] = r2

    return report