from data_utils import DataUtils
import joblib
import math
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from azureml.core import Run
import pandas as pd


class TimeSeriesModeling:
    @staticmethod
    def train_model(train):
        model = auto_arima(train['Sales'], 
                           seasonal=True, 
                           m=7,  # Seasonal periodicity
                           stepwise=True, 
                           suppress_warnings=True, 
                           trace=True)
        return model

    @staticmethod
    def evaluate_model(model, test):
        predictions = model.predict(n_periods=len(test))
        rmse = math.sqrt(mean_squared_error(test['Sales'], predictions))
        return rmse, predictions

    @staticmethod
    def save_and_upload_model(best_model, filename, datastore, target_path="models/"):
        # Save model locally
        model_path = DataUtils.save_model(best_model, "tmp/", filename)
        
        # Upload model to datastore
        DataUtils.upload_to_blob(datastore, "tmp/", target_path)

    @staticmethod
    def run_pipeline(tenant_id: str, config_path: str, data_store_name: str, train_path: str, test_path: str) -> None:

        # Authenticate Workspace
        ws = DataUtils.authenticate_workspace(tenant_id, config_path)

        # Get Datastore
        datastore = DataUtils.get_datastore(ws, data_store_name)

        # Load Data
        train = DataUtils.load_data_from_blob(ws, data_store_name, train_path)
        test = DataUtils.load_data_from_blob(ws, data_store_name, test_path)

        # Run context
        run = Run.get_context()

        # Train Model
        model = TimeSeriesModeling.train_model(train)

        # Evaluate Model
        rmse, _ = TimeSeriesModeling.evaluate_model(model, test)

        # Log model details
        run.log("order", str(model.order))
        run.log("seasonal_order", str(model.seasonal_order))
        run.log("rmse", rmse)

        # Save and upload model 
        if rmse < float("inf"):
            best_model = model
            TimeSeriesModeling.save_and_upload_model(best_model, "best_autoarima_model.pkl", ws.get_default_datastore())

        # Complete the run
        run.complete()
        print("Model Training and Upload Completed")

if __name__ == "__main__":

    tenant_id = "f56f1f69-458e-427b-bada-4cba658f7917"
    config_path = "Users/mypersonall3099/Online_retail/config.json"
    data_store_name = 'workspaceblobstore'
    train_path = 'preprocessed_data/train_data.csv'
    test_path = 'preprocessed_data/test_data.csv'

    TimeSeriesModeling.run_pipeline(
        tenant_id, config_path, data_store_name, train_path, test_path
    )
