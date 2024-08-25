from data_utils import DataUtils
import joblib
import math
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from azureml.core import Run
import pandas as pd
import matplotlib.pyplot as plt


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
        mse = mean_squared_error(test['Sales'], predictions)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(test['Sales'], predictions)
        return rmse, mse, mae, predictions

    @staticmethod
    def save_and_upload_model(best_model, filename, datastore, target_path="models/"):
        # Save model locally
        model_path = DataUtils.save_model(best_model, "tmp/", filename)
        
        # Upload model to datastore
        DataUtils.upload_to_blob(datastore, "tmp/", target_path)

    @staticmethod
    def plot_forecast(test, predictions):
        plt.figure(figsize=(10, 6))
        plt.plot(test['InvoiceDate'], test['Sales'], label='Actual Sales', color='blue')
        plt.plot(test['InvoiceDate'], predictions, label='Predicted Sales', color='red', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title('Actual vs Predicted Sales')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('forecast_plot.png')  # Save the plot locally
        plt.close()  # Close the plot to free up memory
        return 'forecast_plot.png'  # Return the path to the saved plot

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
        rmse, mse, mae, predictions = TimeSeriesModeling.evaluate_model(model, test)

        # Log metrics
        run.log('RMSE', rmse)
        run.log('MSE', mse)
        run.log('MAE', mae)

        # Log model details
        print(f"Order: {model.order}")
        print(f"Seasonal Order: {model.seasonal_order}")
        print(f"RMSE: {rmse}")

        # Plot forecast and upload plot as an artifact
        plot_path = TimeSeriesModeling.plot_forecast(test, predictions)
        run.upload_file('outputs/forecast_plot.png', plot_path)

        # Save and upload model 
        if rmse < float("inf"):
            best_model = model
            TimeSeriesModeling.save_and_upload_model(best_model, "best_autoarima_model.pkl", ws.get_default_datastore())

        # Complete the run
        print("Model Training and Upload Completed")
        run.complete()

if __name__ == "__main__":

    tenant_id = "f56f1f69-458e-427b-bada-4cba658f7917"
    config_path = "Users/mypersonall3099/Online_retail/config.json"
    data_store_name = 'workspaceblobstore'
    train_path = 'preprocessed_data/train_data.csv'
    test_path = 'preprocessed_data/test_data.csv'

    TimeSeriesModeling.run_pipeline(
        tenant_id, config_path, data_store_name, train_path, test_path
    )
