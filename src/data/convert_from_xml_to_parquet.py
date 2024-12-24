import click
import logging
import mlflow
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_xml_to_parquet(xml_path: str, parquet_path: str):
    """
    Convert an XML file to a Parquet file.

    :param xml_path: Path to the input XML file.
    :param parquet_path: Path to the output Parquet file.
    """
    # Read the XML file into a DataFrame
    df = pd.read_xml(xml_path)

    # Log the DataFrame content
    logger.info("XML DataFrame content:\n%s", df)
    mlflow.log_text(df.to_string(), "xml_dataframe_content.txt")

    # Write the DataFrame to a Parquet file
    df.to_parquet(parquet_path)

@click.command()
@click.argument('xml_file_path')
@click.argument('parquet_file_path')
def main(xml_file_path, parquet_file_path):
    """
    Convert an XML file to a Parquet file.

    :param xml_file_path: Path to the input XML file.
    :param parquet_file_path: Path to the output Parquet file.
    """

    convert_xml_to_parquet(xml_file_path, parquet_file_path)

if __name__ == "__main__":
    main()
