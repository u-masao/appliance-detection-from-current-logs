import logging
import pandas as pd
import click
import mlflow

# Configure logging
log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

def merge_parquet_files(parquet_path1, parquet_path2, output_path):
    # Load the Parquet files
    df1 = pd.read_parquet(parquet_path1)
    df2 = pd.read_parquet(parquet_path2)

    # Log the index of the input DataFrames
    logger = logging.getLogger(__name__)
    logger.info("Index of first DataFrame: %s", df1.index)
    logger.info("Index of second DataFrame: %s", df2.index)
    # Rename columns to ensure uniqueness
    df1.columns = [f"env_temp_{col}" for col in df1.columns]
    df2.columns = [f"star_watt_{col}" for col in df2.columns]
    merged_df = pd.concat([df1, df2], axis=1)


    # Save the merged DataFrame to a new Parquet file
    merged_df.to_parquet(output_path)

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
def main(input1, input2, output, mlflow_run_name):
    """
    Merge two Parquet files and save the result.

    :param input1: Path to the first input Parquet file.
    :param input2: Path to the second input Parquet file.
    :param output: Path to the output Parquet file.
    """
    mlflow.set_experiment("merge_datasets")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params({"input1": input1, "input2": input2, "output": output})

    merge_parquet_files(input1, input2, output)

    mlflow.end_run()

if __name__ == "__main__":
    main()
