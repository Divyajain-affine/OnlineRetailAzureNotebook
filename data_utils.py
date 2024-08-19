from azureml.core import Workspace, Datastore, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
import os
import joblib
import pandas as pd

class DataUtils:

    #-------------Authenticate Azure ML Workspace----------------
    @staticmethod
    def authenticate_workspace(tenant_id: str, config_path: str) -> Workspace:
        interactive_auth = InteractiveLoginAuthentication(tenant_id=tenant_id, force=True)
        ws = Workspace.from_config(config_path)
        return ws
    
    #-------------Retrieve the Datastore from the Workspace--------------

    @staticmethod
    def get_datastore(ws: Workspace, datastore_name: str) -> Datastore:
        return Datastore.get(ws, datastore_name)

    #------------Load the dataset from Azure Blob Storage-----------------

    @staticmethod
    def load_data_from_blob(ws: Workspace, data_store_name: str, file_path: str) -> pd.DataFrame:
        datastore = Datastore.get(ws, data_store_name)
        dataset = Dataset.Tabular.from_delimited_files(path=[(datastore, file_path)])
        return dataset.to_pandas_dataframe()

    #----------------Save the DataFrame to a specified directory----------

    @staticmethod
    def save_data(df, dir_path: str, file_name: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        df.to_csv(file_path)
        return file_path

    #-----------Save the model to a specified directory--------------------

    @staticmethod
    def save_model(model, dir_path: str, file_name: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, file_name)
        joblib.dump(model, file_path)
        return file_path

    #--------------Upload files(data or models) to Azure Blob Storage------

    @staticmethod
    def upload_to_blob(datastore: Datastore, src_dir: str, target_path: str) -> None:
        if not os.path.exists(src_dir):
            raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

        datastore.upload(src_dir=src_dir, target_path=target_path, overwrite=True)
        print(f"Uploaded files from {src_dir} to {target_path} in datastore.")

