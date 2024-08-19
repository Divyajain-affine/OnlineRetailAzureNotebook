
from data_utils import DataUtils
import os
from azureml.core import Run
import pandas as pd

class DataWrangling:

    @staticmethod
    def run_wrangling_pipeline(tenant_id: str, config_path: str, data_store_name: str, raw_data_file_path: str) -> None:
        
        # Step 1: Authenticate Workspace using shared utility
        ws = DataUtils.authenticate_workspace(tenant_id, config_path)

        # Step 2: Load Data using shared utility
        df = DataUtils.load_data_from_blob(ws, data_store_name, raw_data_file_path)

        # Step 3: Save the wrangled data locally using shared utility
        temp_path = DataUtils.save_data(df, "tmp/", "wrangled.csv")

        # Step 4: Upload the wrangled data back to Blob Storage using shared utility
        datastore = ws.get_default_datastore()
        DataUtils.upload_to_blob(datastore, "tmp/", "data_wrangling")

        print("Completed Wrangling Process!")

if __name__ == "__main__":
    tenant_id = "f56f1f69-458e-427b-bada-4cba658f7917"
    config_path = "Users/mypersonall3099/Online_retail/config.json"
    data_store_name = 'workspaceblobstore'
    raw_data_file_path = 'online_retail_utf8.csv'

    DataWrangling.run_wrangling_pipeline(
        tenant_id, config_path, data_store_name, raw_data_file_path
    )
