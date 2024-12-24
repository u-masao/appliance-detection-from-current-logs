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
    # Add a column for days since the start of the year
    df["days_since_year_start"] = df.index.dayofyear
    # Add columns for month, hour, and minute
    df["month"] = df.index.month
    df["hour"] = df.index.hour
    df["minute"] = df.index.minute
    df["watt_black_minus_kitchen"] = (
        df["star_watt_sensor0"] - df["star_watt_sensor2"]
    )
    df["watt_red_minus_living"] = (
        df["star_watt_sensor1"] - df["star_watt_sensor3"]
    )
    df["watt_total"] = df["star_watt_sensor0"] - df["star_watt_sensor1"]
    df = df.rename(
        columns={
            "star_watt_sensor0": "watt_black",
            "star_watt_sensor1": "watt_red",
            "star_watt_sensor2": "watt_kitchen",
            "star_watt_sensor3": "watt_living",
            "env_temp_sensor0": "temperature_outside",
        }
    )
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
    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    logger.info(f"Input path: {input_parquet_path}")
    logger.info(f"Output path: {output_parquet_path}")

    mlflow.set_experiment("build_features")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params(
        {
            "input_parquet_path": input_parquet_path,
            "output_parquet_path": output_parquet_path,
            "mlflow_run_name": mlflow_run_name,
        }
    )

    # Load the Parquet file
    df = pd.read_parquet(input_parquet_path)
    logger.info("Loaded DataFrame:\n%s", df.head())

    # Create features
    df = create_features(df)
    logger.info("DataFrame with features:\n%s", df.head())

    # Log the entire DataFrame with features
    logger.info("Resulting DataFrame:\n%s", df)
    logger.info("Output DataFrame columns: %s", df.columns)
    df.to_parquet(output_parquet_path)

    mlflow.end_run()
    logger.info("==== end process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()