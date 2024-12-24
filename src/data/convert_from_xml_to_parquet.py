import pandas as pd

def convert_xml_to_parquet(xml_path: str, parquet_path: str):
    """
    Convert an XML file to a Parquet file.

    :param xml_path: Path to the input XML file.
    :param parquet_path: Path to the output Parquet file.
    """
    # Read the XML file into a DataFrame
    df = pd.read_xml(xml_path)

    # Write the DataFrame to a Parquet file
    df.to_parquet(parquet_path)

if __name__ == "__main__":
    # Define the input and output paths
    xml_file_path = "data/raw/env_temp.xml"
    parquet_file_path = "data/interim/env_temp.parquet"

    # Convert the XML to Parquet
    convert_xml_to_parquet(xml_file_path, parquet_file_path)
