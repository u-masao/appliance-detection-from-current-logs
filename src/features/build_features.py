import logging

import click
import mlflow
import pandas as pd


def create_features(df):
    """
    Create features for time series data.

    :param df: Input DataFrame.
    :return: DataFrame with new features.
    """
    df = df.rename(
        columns={
            "star_watt_sensor0": "watt_black",
            "star_watt_sensor1": "watt_red",
            "star_watt_sensor2": "watt_living",
            "star_watt_sensor3": "watt_kitchen",
            "env_temp_sensor0": "temperature_outside",
        }
    )
    df["watt_black_minus_living"] = df["watt_black"] - df["watt_living"]
    df["watt_red_minus_kitchen"] = df["watt_red"] - df["watt_kitchen"]
    df["watt_total"] = df["watt_black"] + df["watt_red"]

    # Add a column for days since the start of the year
    jst_dt = df.index.tz_convert("Asia/Tokyo")
    df["days_since_year_start"] = jst_dt.dayofyear
    df["day_of_week"] = jst_dt.dayofweek
    # Add columns for month, hour, and minute
    df["month"] = jst_dt.month
    df["hour"] = jst_dt.hour
    df["minute"] = jst_dt.minute
    return df


@click.command()
@click.argument("input_parquet_path", type=click.Path(exists=True))
@click.argument(
    "output_parquet_path",
    type=click.Path(),
    default="data/interim/features.parquet",
)
@click.option(
    "--mlflow_run_name",
    type=str,
    default="build_features",
    help="Name of the MLflow run.",
)
def main(input_parquet_path, output_parquet_path, mlflow_run_name):
    """
    Build features for time series data.

    :param input_parquet_path: Path to the input Parquet file.
    :param output_parquet_path: Path to the output Parquet file.
    """

    # init log
    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    mlflow.set_experiment("build_features")
    mlflow.start_run(run_name=mlflow_run_name)

    # log args
    cli_args = {
        "input_parquet_path": input_parquet_path,
        "output_parquet_path": output_parquet_path,
        "mlflow_run_name": mlflow_run_name,
    }
    logger.info(f"{cli_args=}")
    mlflow.log_params({f"args.{k}": v for k, v in cli_args.items()})

    # Load the Parquet file
    input_df = pd.read_parquet(input_parquet_path)
    logger.info("Loaded DataFrame:\n%s", input_df.head())

    # Create features
    df = create_features(input_df)

    # Log the entire DataFrame with features
    logger.info("DataFrame with features:\n%s", df.head())
    logger.info("Output DataFrame columns: %s", df.columns)

    # output dataframe
    df.to_parquet(output_parquet_path)

    # log
    log_params = {
        "input.length": input_df.shape[0],
        "input.columns": input_df.shape[1],
        "output.length": df.shape[0],
        "output.columns": df.shape[1],
    }
    logger.info(f"{log_params=}")
    mlflow.log_params(log_params)
    logger.info("==== end process ====")

    # cleanup
    mlflow.end_run()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
