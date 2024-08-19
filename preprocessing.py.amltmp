from data_utils import DataUtils
from azureml.core import Datastore
import pandas as pd
from azureml.core import Run


class DataPreprocessing:

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

        df = df[['Quantity', 'Price', 'InvoiceDate']].drop_duplicates()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df = df.dropna()
        df["Sales"] = df["Quantity"] * df["Price"]

        # For daily sales
        df.set_index('InvoiceDate', inplace=True)
        df_daily = df.resample('D').sum()
        df_daily = df_daily[['Sales']]
        
        incremental_data = df_daily.tail(10)
        remaining_data = df_daily.iloc[:-10]

        return incremental_data, remaining_data
    
    @staticmethod
    def split_train_test(data: pd.DataFrame, train_ratio: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
        train_size = int(len(data) * train_ratio)
        train = data[:train_size]
        test = data[train_size:]
        return train, test

    @staticmethod
    def run_preprocessing_pipeline(tenant_id: str, config_path: str, data_store_name: str, wrangled_file_path: str) -> None:

        # Step 1: Authenticate Workspace using shared utility
        ws = DataUtils.authenticate_workspace(tenant_id, config_path)

        # Step 2: Load Data using shared utility
        df = DataUtils.load_data_from_blob(ws, data_store_name, wrangled_file_path)
        
        # Step 3: Preprocess Data
        incremental_data, remaining_data = DataPreprocessing.preprocess_data(df)

        # Step 4: Save Incremental Data using shared utility
        incremental_path = "incremental_data/"
        DataUtils.save_data(incremental_data, incremental_path, "incremental_data.csv")

        # Step 5: Split Remaining Data into Train and Test
        train, test = DataPreprocessing.split_train_test(remaining_data)

        # Step 6: Save Train and Test Data using shared utility
        output_path = "preprocessed_data/"
        DataUtils.save_data(train, output_path, "train_data.csv")
        DataUtils.save_data(test, output_path, "test_data.csv")

        # Step 7: Upload Data Back to Blob Storage using shared utility
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
