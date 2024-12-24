import logging

import click
import mlflow
import pandas as pd


def convert_xml_to_parquet(xml_path: str, parquet_path: str):
    """
    Convert an XML file to a Parquet file.

    :param xml_path: Path to the input XML file.
    :param parquet_path: Path to the output Parquet file.
    """
    logger = logging.getLogger(__name__)
    # Read the XML file into a DataFrame
    df = pd.read_xml(xml_path)

    # Log the DataFrame content
    logger.info("XML DataFrame content:\n%s", df)
    mlflow.log_text(df.to_string(), "xml_dataframe_content.txt")

    # Log the number of rows and columns
    num_rows, num_cols = df.shape
    logger.info(
        "Number of rows: %d, Number of columns: %d", num_rows, num_cols
    )
    mlflow.log_param("num_rows", num_rows)
    mlflow.log_param("num_cols", num_cols)

    # Write the DataFrame to a Parquet file
    df.to_parquet(parquet_path)


@click.command()
@click.argument("xml_file_path", type=click.Path(exists=True))
@click.argument("parquet_file_path", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """
    Convert an XML file to a Parquet file.

    :param xml_file_path: Path to the input XML file.
    :param parquet_file_path: Path to the output Parquet file.
    """

    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    logger.info({f"args.{k}": v for k, v in kwargs.items()})

    mlflow.set_experiment("make_dataset")
    mlflow.start_run(run_name=kwargs["mlflow_run_name"])
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    convert_xml_to_parquet(
        kwargs["xml_file_path"], kwargs["parquet_file_path"]
    )

    mlflow.end_run()
    logger.info("==== end process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
