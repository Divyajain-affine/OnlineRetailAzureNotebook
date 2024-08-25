from data_utils import DataUtils  
import pandas as pd  
from statsmodels.tsa.stattools import adfuller  
from azureml.core import Datastore  
from azureml.core import Run  


class DataPreprocessing:

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
 
        # Set index and resample
        df = df.set_index('InvoiceDate').resample('D').sum().fillna(0).reset_index()
 
        # Check for NaNs or Infs and handle them
        if df.isnull().values.any() or (df == float('inf')).values.any() 
        or (df == float('-inf')).values.any():
             # Replace inf values with NA
            df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)  
            df.fillna(0, inplace=True)  # Replace NaNs with 0

        # Create daily sales DataFrame
        df_daily = df[['InvoiceDate', 'Sales']]

        # Create incremental data (last 10 rows) and remaining data
        incremental_data = df_daily.tail(10)
        remaining_data = df_daily.iloc[:-10]
        return incremental_data, remaining_data

    @staticmethod
    def split_train_test(data: pd.DataFrame, train_ratio: float = 0.8) -> 
    (pd.DataFrame, pd.DataFrame):
        train_size = int(len(data) * train_ratio)
        train = data[:train_size]
        test = data[train_size:]
        return train, test

    @staticmethod
    def run_preprocessing_pipeline(tenant_id: str, config_path: str, data_store_name: str, wrangled_file_path: str) -> None:
        ws = DataUtils.authenticate_workspace(tenant_id, config_path)
        df = DataUtils.load_data_from_blob(ws, data_store_name, wrangled_file_path)
        incremental_data, remaining_data = DataPreprocessing.preprocess_data(df)
        incremental_path = "incremental_data/"
        DataUtils.save_data(incremental_data, incremental_path, "incremental_data.csv")
        train, test = DataPreprocessing.split_train_test(remaining_data)
        output_path = "preprocessed_data/"
        DataUtils.save_data(train, output_path, "train_data.csv")
        DataUtils.save_data(test, output_path, "test_data.csv")
        datastore = Datastore.get(ws, data_store_name)
        DataUtils.upload_to_blob(datastore, incremental_path, "incremental_data")
        DataUtils.upload_to_blob(datastore, output_path, "preprocessed_data")
        print("Data Preprocessing and Export Completed")

if __name__ == "__main__":
    tenant_id = "f56f1f69-458e-427b-bada-4cba658f7917"
    config_path = "Users/mypersonall3099/Online_retail/config.json"
    data_store_name = 'workspaceblobstore'
    wrangled_file_path = 'data_wrangling/wrangled.csv'

    DataPreprocessing.run_preprocessing_pipeline(
        tenant_id, config_path, data_store_name, wrangled_file_path
    )
