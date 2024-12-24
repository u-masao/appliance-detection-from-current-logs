import logging

import click
import mlflow
import pandas as pd


def align_intervals(df1, df2, ffill_multiplier):
    """
    Align the intervals of two DataFrames by resampling to the shorter interval.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :return: Tuple of resampled DataFrames.
    """
    # Calculate time intervals
    interval_df1 = df1.index.to_series().diff().min()
    interval_df2 = df2.index.to_series().diff().min()

    # Determine the shorter interval
    target_interval = min(interval_df1, interval_df2)

    # Calculate the limit for ffill based on the intervals
    ffill_limit = ffill_multiplier * (
        max(interval_df1, interval_df2) // min(interval_df1, interval_df2)
    )
    if interval_df1 > interval_df2:
        df1 = df1.resample(target_interval).ffill(limit=ffill_limit)
    elif interval_df2 > interval_df1:
        df2 = df2.resample(target_interval).ffill(limit=ffill_limit)

    return df1, df2


def concat_dataframes(df1, df2):
    """
    Concatenate two DataFrames with prefixed column names.

    :param df1: First DataFrame.
    :param df2: Second DataFrame.
    :return: Concatenated DataFrame.
    """
    df1.columns = [f"env_temp_{col}" for col in df1.columns]
    df2.columns = [f"star_watt_{col}" for col in df2.columns]
    return pd.concat([df1, df2], axis=1)


def remove_null_rows(df, window="6H"):
    """
    Remove rows with null values and those within a specified window of them.

    :param df: DataFrame to process.
    :param window: Time window to consider for dropping rows.
    :return: DataFrame with specified rows removed.
    """
    null_mask = df.isnull().any(axis=1)
    drop_mask = (
        null_mask.rolling(window=window, min_periods=1).max().astype(bool)
    )
    return df[~drop_mask]


def add_gap_column(df):
    """
    Add a gap column to indicate time continuity breaks.

    :param df: DataFrame to process.
    :return: DataFrame with a gap column added.
    """
    time_diffs = df.index.to_series().diff().fillna(pd.Timedelta(seconds=0))
    min_diff = time_diffs[time_diffs > pd.Timedelta(seconds=0)].min()
    gap = (time_diffs != min_diff) | (time_diffs.shift(-1) != min_diff)
    gap.iloc[0] = True  # First row
    gap.iloc[-1] = True  # Last row
    df["gap"] = gap
    return df


def merge_parquet_files(df1, df2, ffill_multiplier, window):

    logger = logging.getLogger(__name__)
    df1, df2 = align_intervals(df1, df2, ffill_multiplier)

    merged_df = concat_dataframes(df1, df2)

    # Remove rows with null values and those within 6 hours of them
    merged_df = remove_null_rows(merged_df, window=window)
    merged_df = add_gap_column(merged_df)

    # Log the time differences between indices
    time_diffs = merged_df.index.to_series().diff().value_counts().sort_index()
    logger.info("Time differences between indices:\n%s", time_diffs)

    # Add the gap column
    null_counts = merged_df.isnull().sum()
    logger.info("Null counts per column:\n%s", null_counts)

    return merged_df


@click.command()
@click.argument("input1", type=click.Path(exists=True))
@click.argument("input2", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--mlflow_run_name",
    type=str,
    default="merge_parquet_files",
    help="Name of the MLflow run.",
)
@click.option(
    "--ffill_multiplier",
    type=int,
    default=4,
    help="Multiplier for the forward fill limit.",
)
@click.option(
    "--window",
    type=str,
    default="6H",
    help="Time window for dropping rows with null values.",
)
def main(input1, input2, output, mlflow_run_name, ffill_multiplier, window):
    """
    Merge two Parquet files and save the result.

    :param input1: Path to the first input Parquet file.
    :param input2: Path to the second input Parquet file.
    :param output: Path to the output Parquet file.
    """
    mlflow.set_experiment("merge_datasets")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params(
        {
            "input1": input1,
            "input2": input2,
            "output": output,
            "mlflow_run_name": mlflow_run_name,
            "ffill_multiplier": ffill_multiplier,
            "window": window,
        }
    )

    logger = logging.getLogger(__name__)

    # Load the Parquet files
    df1 = pd.read_parquet(input1)
    df2 = pd.read_parquet(input2)

    # Log the index of the input DataFrames
    logger.info("Index of first DataFrame: %s", df1.index)
    logger.info("Index of second DataFrame: %s", df2.index)

    merged_df = merge_parquet_files(df1, df2, ffill_multiplier, window)

    # Save the merged DataFrame to a Parquet file
    merged_df.to_parquet(output)

    mlflow.end_run()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
