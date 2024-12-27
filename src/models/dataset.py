import logging

import pandas as pd
import torch
from torch.utils.data import Dataset


# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length, output_length, target_columns):
        self.target_columns = target_columns
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __init__(self, data, input_length, output_length, target_columns):
        self.target_columns = target_columns
        self.data = data
        self.input_length = input_length
        self.output_length = output_length
        self.valid_indices = self._compute_valid_indices()

    def _compute_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.data) - self.input_length - self.output_length):
            xy_df = self.data.iloc[idx : idx + self.input_length + self.output_length]
            if xy_df["gap"].sum() == 0:
                valid_indices.append(idx)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]
        x_df = self.data.iloc[idx : idx + self.input_length]
        x = x_df.drop("gap", axis=1).to_numpy(dtype="float32")
        y = self.data.iloc[
            idx + self.input_length : idx + self.input_length + self.output_length
        ][self.target_columns].to_numpy(dtype="float32")
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_data(file_path, fraction=1.0):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.debug(f"Data types:\n{df.dtypes}")
    if fraction < 1.0:
        df = df.iloc[: int(len(df) * fraction)]
        logger.info(
            f"Data reduced to {len(df)} samples for development (sequentially)"
        )
    logger.info("Data loaded successfully")
    return df
