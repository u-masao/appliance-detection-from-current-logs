import logging
import click
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
    "--input_length",
    type=int,
    default=0,
    help="Number of rows to skip between splits.",
)
def main(input_path, output_train_path, output_val_path, output_test_path, train_ratio, val_ratio, input_length):
    logger = logging.getLogger(__name__)
    logger.info("==== start data splitting ====")
    logger.info(f"Input path: {input_path}")

    df = pd.read_parquet(input_path)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size + input_length : train_size + input_length + val_size]
    test_df = df.iloc[train_size + input_length + val_size + input_length :]

    train_df.to_parquet(output_train_path)
    val_df.to_parquet(output_val_path)
    test_df.to_parquet(output_test_path)

    logger.info("Data splitting completed")
    logger.info("==== end data splitting ====")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
