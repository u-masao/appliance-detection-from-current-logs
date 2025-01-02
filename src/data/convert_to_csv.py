import logging

import click
import mlflow

from src.models.train_model import load_data


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
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
    output_path,
    data_fraction,
    mlflow_run_name,
):
    logger = logging.getLogger(__name__)
    logger.info("==== start inference process ====")
    logger.info(f"Input path: {input_path}")

    mlflow.set_experiment("data")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params(
        {
            "input_path": input_path,
            "output_path": output_path,
        }
    )

    # Load data
    df = load_data(input_path, fraction=data_fraction)

    # output data
    df.to_csv(output_path)

    mlflow.end_run()
    logger.info("==== end inference process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
