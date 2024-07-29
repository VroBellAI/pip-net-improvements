import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Sequence


class Logger:
    def __init__(
        self,
        log_dir: str,
        epoch_log_file_name: str = "log_epoch_overview.csv",
        step_log_file_name: str = "log_step_overview.csv",
    ):
        # Set log file path;
        self.log_dir = log_dir

        self.epoch_log_file_name = epoch_log_file_name
        self.step_log_file_name = step_log_file_name

        self.epoch_log_file_path = f"{self.log_dir}/{self.epoch_log_file_name}"
        self.step_log_file_path = f"{self.log_dir}/{self.step_log_file_name}"

        # Ensure the directories exist;
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.isdir(self.metadata_dir):
            os.mkdir(self.metadata_dir)

        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        # Load existing log, or create new one;
        self.epoch_log_df = self.load_or_create_log_file(
            file_path=self.epoch_log_file_path
        )
        self.step_log_df = self.load_or_create_log_file(
            file_path=self.step_log_file_path
        )

    @property
    def checkpoint_dir(self):
        return self.log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self.log_dir + '/metadata'

    def log_epoch_info(self, values: Dict[str, Any]):
        """
        Adds new values to epoch DataFrame,
        saves the result to CSV file.
        """
        # Add values to current log data frame;
        self.epoch_log_df = self.log_values_to_df(
            values=values,
            log_df=self.epoch_log_df,
        )

        # Save result to file;
        self.epoch_log_df.to_csv(self.epoch_log_file_path, index=False)

    def log_step_info(self, values: Dict[str, Any]):
        """
        Adds new values to step DataFrame,
        saves the result to CSV file.
        """
        # Add values to current log data frame;
        self.step_log_df = self.log_values_to_df(
            values=values,
            log_df=self.step_log_df,
        )

        # Save result to file;
        self.step_log_df.to_csv(self.step_log_file_path, index=False)

    @staticmethod
    def log_values_to_df(values: Dict[str, Any], log_df: pd.DataFrame):
        # Convert the input dictionary to a DataFrame;
        new_row = pd.DataFrame([values])

        # Append the new row to the existing DataFrame;
        log_df = pd.concat([log_df, new_row], ignore_index=True)

        # Ensure all columns are present and fill missing values with None;
        all_columns = log_df.columns.union(new_row.columns)
        log_df = log_df.reindex(
            columns=all_columns,
            fill_value=None,
        )
        return log_df

    @staticmethod
    def load_or_create_log_file(file_path: str) -> pd.DataFrame:
        """
        Loads logging csv file if exists.
        If not, creates the new one.
        """
        if os.path.exists(file_path):
            return pd.read_csv(file_path)

        return pd.DataFrame()

