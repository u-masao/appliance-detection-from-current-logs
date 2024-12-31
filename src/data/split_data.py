import logging

import click
import mlflow
import pandas as pd


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_train_path", type=click.Path())
@click.argument("output_val_path", type=click.Path())
@click.argument("output_test_path", type=click.Path())
@click.option(
    "--train_ratio",
    type=float,
    default=0.7,
    help="Ratio of data to use for training.",
)
@click.option(
    "--val_ratio",
    type=float,
    default=0.15,
    help="Ratio of data to use for validation.",
)
@click.option(
    "--input_sequence_length",
    type=int,
    default=0,
    help="Number of rows to skip between splits.",
)
def main(
    input_path,
    output_train_path,
    output_val_path,
    output_test_path,
    train_ratio,
    val_ratio,
    input_sequence_length,
):
    logger = logging.getLogger(__name__)
    mlflow.set_experiment("make_dataset")
    mlflow.start_run()
    logger.info("==== start data splitting ====")
    logger.info(f"Input path: {input_path}")

    df = pd.read_parquet(input_path)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[
        train_size
        + input_sequence_length : train_size
        + input_sequence_length
        + val_size
    ]
    test_df = df.iloc[
        train_size + input_sequence_length + val_size + input_sequence_length :
    ]

    train_df.to_parquet(output_train_path)
    train_start, train_end = train_df.index.min(), train_df.index.max()
    logger.info(f"Train data period: {train_start} to {train_end}")
    mlflow.log_param("train_data_period", f"{train_start} to {train_end}")
    logger.info(f"Train data shape: {train_df.shape}")
    mlflow.log_param("train_data_shape", train_df.shape)
    val_df.to_parquet(output_val_path)
    val_start, val_end = val_df.index.min(), val_df.index.max()
    logger.info(f"Validation data period: {val_start} to {val_end}")
    mlflow.log_param("val_data_period", f"{val_start} to {val_end}")
    logger.info(f"Validation data shape: {val_df.shape}")
    mlflow.log_param("val_data_shape", val_df.shape)
    test_df.to_parquet(output_test_path)
    test_start, test_end = test_df.index.min(), test_df.index.max()
    logger.info(f"Test data period: {test_start} to {test_end}")
    mlflow.log_param("test_data_period", f"{test_start} to {test_end}")
    logger.info(f"Test data shape: {test_df.shape}")
    mlflow.log_param("test_data_shape", test_df.shape)

    logger.info("Data splitting completed")
    logger.info("==== end data splitting ====")
    mlflow.end_run()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
