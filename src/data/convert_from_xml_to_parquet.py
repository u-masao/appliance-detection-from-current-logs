import logging
import xml.etree.ElementTree as ET

import click
import mlflow
import pandas as pd


def create_timeseries_index(step_seconds, end_time, length):
    """
    pandas timeseries index を作成する関数

    Args:
      step_seconds: ステップ秒数
      end_time: 最後の時刻 (UNIXEPOCHタイム)
      length: 長さ

    Returns:
      pandas.DatetimeIndex: 作成されたtimeseries index
    """
    end_time = pd.to_datetime(end_time, unit="s", utc=True)
    start_time = end_time - pd.Timedelta(seconds=(length - 1) * step_seconds)
    return pd.date_range(
        start=start_time, end=end_time, freq=f"{step_seconds}S"
    )


def parse_database(root, step: int, quantized_ts: int, cf_text: str):
    logger = logging.getLogger(__name__)
    # "LAST" のCF要素を特定し、その次の兄弟要素である <database> から <row> 要素を取得
    rows = None
    for rra in root.findall(".//rra"):
        cf = rra.find("cf")
        if cf_text in cf.text:
            rows = rra.find("database").findall("row")
            break  # "LAST" のCF要素が見つかったらループを抜ける

    if rows is None:
        raise RuntimeError("データが取得できませんでした")

    data = [[float(v.text) for v in row.findall("v")] for row in rows]
    ts_index = create_timeseries_index(step, quantized_ts, len(data))
    logger.info(f"{ts_index=}")
    df = pd.DataFrame(data, index=ts_index)
    df.columns = [f"sensor{i}" for i in range(df.shape[1])]
    return df


def convert_xml_to_parquet(xml_path: str, parquet_path: str, cf_text: str):
    """
    Convert an XML file to a Parquet file.

    :param xml_path: Path to the input XML file.
    :param parquet_path: Path to the output Parquet file.
    """

    # init logger
    logger = logging.getLogger(__name__)

    # load data
    tree = ET.parse(xml_path)

    # parse data
    root = tree.getroot()
    step = int(root.find("step").text)
    last_update = int(root.find("lastupdate").text)
    quantized_ts = step * (last_update // step)
    df = parse_database(root, step, quantized_ts, cf_text)

    # output
    logger.info(f"{step=}")
    logger.info(f"{last_update=}")
    logger.info(f"{quantized_ts=}")
    logger.info("XML DataFrame content:\n%s", df)

    # Log the number of rows and columns
    num_rows, num_cols = df.shape
    logger.info(
        "Number of rows: %d, Number of columns: %d", num_rows, num_cols
    )
    mlflow.log_param("output.rows", num_rows)
    mlflow.log_param("output.cols", num_cols)

    # Write the DataFrame to a Parquet file
    df.to_parquet(parquet_path)


@click.command()
@click.argument("xml_file_path", type=click.Path(exists=True))
@click.argument("parquet_file_path", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop", help="Name of the MLflow run.")
@click.option("--cf_text", type=str, default="LAST", help="CF text to search for in the XML.")
def main(xml_file_path, parquet_file_path, mlflow_run_name, cf_text):
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

    convert_xml_to_parquet(xml_file_path, parquet_file_path, cf_text)

    mlflow.end_run()
    logger.info("==== end process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
