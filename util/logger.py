import os
import pandas as pd
from typing import Dict, Any


class Logger:
    def __init__(
        self,
        log_dir: str,
        log_file_name: str = "log_epoch_overview",
    ):
        # Set log file path;
        self.log_dir = log_dir
        self.log_file_name = f"{log_file_name}.csv"
        self.log_file_path = f"{self.log_dir}/{self.log_file_name}"

        # Ensure the directories exist;
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.isdir(self.metadata_dir):
            os.mkdir(self.metadata_dir)

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        # Load existing log, or create new one;
        self.log_df = self.load_or_create_log_file()

    @property
    def checkpoint_dir(self):
        return self.log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self.log_dir + '/metadata'

    def log_epoch_info(self, values: Dict[str, Any]):
        # Convert the input dictionary to a DataFrame;
        new_row = pd.DataFrame([values])

        # Append the new row to the existing DataFrame;
        self.log_df = pd.concat([self.log_df, new_row], ignore_index=True)

        # Ensure all columns are present and fill missing values with None;
        all_columns = self.log_df.columns.union(new_row.columns)
        self.log_df = self.log_df.reindex(
            columns=all_columns,
            fill_value=None,
        )

        # Save the updated DataFrame to the CSV file;
        self.log_df.to_csv(self.log_file_path, index=False)

    def load_or_create_log_file(self) -> pd.DataFrame:
        """
        Loads logging csv file if exists.
        If not, creates the new one.
        """
        if os.path.exists(self.log_file_name):
            return pd.read_csv(self.log_file_name)

        return pd.DataFrame()
