import logging

import click
import mlflow
import pandas as pd

from src.models.train_model import load_data


def df_to_monthly_csv(df, output_dir):
    """
    pandas DataFrame の Index に datetime を設定したデータを受け取り、
    1ヶ月毎にCSVファイルとして出力する関数。

    Args:
        df (pd.DataFrame):
            Index に datetime を設定した pandas DataFrame。
        output_dir (str, optional):
            出力ファイル名のプレフィックス。Defaults to "monthly_data".
    """

    # indexの最小値と最大値から年と月の範囲を取得
    start_year = df.index.min().year
    end_year = df.index.max().year

    # 1ヶ月ごとにCSVファイルに出力
    for year in range(start_year, end_year + 1):  # 最小年〜最大年を処理
        for month in range(1, 13):  # 1月から12月を処理
            start_date = pd.to_datetime(f"{year}-{month:02d}-01")
            end_date = (
                start_date + pd.DateOffset(months=1) - pd.DateOffset(days=1)
            )

            # 1ヶ月分のデータを取得
            monthly_data = df[
                (df.index >= start_date) & (df.index <= end_date)
            ]

            # データが存在する場合のみCSVファイルに出力
            if not monthly_data.empty:
                # CSVファイル名を作成 (例: monthly_data_2022-01.csv)
                file_name = f"{output_dir}/{year}-{month:02d}.csv"

                # CSVファイルに出力
                monthly_data.to_csv(file_name)


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--split",
    type=str,
    default="monthly",
    help="split period",
)
@click.option(
    "--data_fraction",
    type=float,
    default=1.0,
    help="Fraction of data to load for testing.",
)
@click.option(
    "--mlflow_run_name",
    type=str,
    default="inference_run",
    help="Name of the MLflow run.",
)
def main(
    input_path,
    output_dir,
    data_fraction,
    mlflow_run_name,
):
    logger = logging.getLogger(__name__)
    logger.info("==== start inference process ====")
    logger.info(f"Input path: {input_path}")

    mlflow.set_experiment("make_dataset")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params(
        {
            "input_path": input_path,
            "output_dir": output_dir,
        }
    )

    # Load data
    df = load_data(input_path, fraction=data_fraction)

    df.index = df.index.tz_convert("Asia/Tokyo")
    df.index.name = "timestamp_jst"

    # output data
    df_to_monthly_csv(df, f"{output_dir}")

    mlflow.end_run()
    logger.info("==== end inference process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
