import joblib
import json
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model_3
    model_3_path = Model.get_model_path(model_name='best_autoarima_model')
    model_3 = joblib.load(model_3_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        
        # Convert the JSON data to a DataFrame
        data_df = pd.DataFrame(data, columns=['InvoiceDate', 'Sales'])
        
        # Extract the 'Sales' column for prediction
        sales_data = data_df['Sales'].values
        
        # Define how many future periods you want to predict
        n_periods = 10  # Change this to the number of future periods you want to forecast
        
        # Make predictions
        result_1 = model_3.predict(n_periods=n_periods)
        
        return {"predictions": result_1.tolist()}
    except Exception as e:
        result = str(e)
        return {"error": result}


