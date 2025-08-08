from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental level of education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test preparation course'),
                math_score=int(request.form.get('math score')),
                reading_score=int(request.form.get('reading score')),
                writing_score=int(request.form.get('writing score'))
            )
            pred_df = data.get_data_as_data_frame()
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            return render_template('home.html', results=results[0])
        except Exception as e:
            return f"❌ Error occurred: {e}"
    return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")