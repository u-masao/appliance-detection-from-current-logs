import torch
from torch.utils.data import IterableDataset


# Custom Dataset
class TimeSeriesDataset(IterableDataset):
    def __init__(self, data, input_length, output_length, target_columns):
        self.target_columns = target_columns
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __iter__(self):
        for idx in range(
            len(self.data) - self.input_length - self.output_length
        ):
            x_df = self.data.iloc[idx : idx + self.input_length]
            if x_df["gap"].sum() > 0:
                continue  # Skip this data if there is a gap
            x = x_df.drop("gap", axis=1).to_numpy(dtype="float32").flatten()
            y = (
                self.data.iloc[
                    idx
                    + self.input_length : idx
                    + self.input_length
                    + self.output_length
                ][self.target_columns]
                .to_numpy(dtype="float32")
                .flatten()
            )
            yield torch.tensor(x, dtype=torch.float32), torch.tensor(
                y, dtype=torch.float32
            )
import logging
import pandas as pd

def load_data(file_path, fraction=1.0):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Data types:\n{df.dtypes}")
    if fraction < 1.0:
        df = df.iloc[: int(len(df) * fraction)]
        logger.info(
            f"Data reduced to {len(df)} samples for development (sequentially)"
        )
    logger.info("Data loaded successfully")
    return df
