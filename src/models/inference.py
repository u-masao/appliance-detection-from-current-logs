import logging
import math

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from src.models.model import TransformerModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.train_model import load_data

from src.models.dataset import TimeSeriesDataset


def run_inference(
    model,
    test_df,
    input_length,
    output_length,
    target_columns,
    batch_size,
    device,
):
    # Create test dataset and loader
    test_dataset = TimeSeriesDataset(
        test_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    model.eval()
    trains = []
    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference [Test]"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            trains.append(x.cpu().numpy())
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())

    # Convert lists to arrays
    trains = np.concatenate(trains, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    train_df = pd.DataFrame(trains).add_prefix("train_")
    pred_df = pd.DataFrame(predictions).add_prefix("pred_")
    actual_df = pd.DataFrame(actuals).add_prefix("actual_")
    concat_df = pd.concat([train_df, pred_df, actual_df], axis=1)

    return concat_df


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--input_length",
    type=int,
    default=60 * 3,
    help="Input length for the time series data.",
)
@click.option(
    "--output_length",
    type=int,
    default=5,
    help="Output length for the time series data.",
)
@click.option(
    "--data_fraction",
    type=float,
    default=1.0,
    help="Fraction of data to load for testing.",
)
@click.option(
    "--batch_size", type=int, default=32, help="Batch size for inference."
)
@click.option(
    "--device",
    type=str,
    default="cpu",
    help="Device to run inference on (cpu or cuda).",
)
@click.option(
    "--mlflow_run_name",
    type=str,
    default="inference_run",
    help="Name of the MLflow run.",
)
def main(
    model_path,
    input_path,
    output_path,
    data_fraction,
    input_length,
    output_length,
    batch_size,
    device,
    mlflow_run_name,
):
    logger = logging.getLogger(__name__)
    logger.info("==== start inference process ====")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input path: {input_path}")

    mlflow.set_experiment("inference")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params({"model_path": model_path, "input_path": input_path})
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]

    # Load model configuration
    try:
        model_config = torch.load(
            model_path.replace(".pth", "_config.pth"), weights_only=True
        )
    except FileNotFoundError:
        raise RuntimeError(
            "Model configuration file not found. Ensure the model was trained with the configuration saved."
        )
    model = TransformerModel(**model_config)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.to(device)

    # Load and split data
    df = load_data(input_path, fraction=data_fraction)

    # Run inference
    output_df = run_inference(
        model=model,
        test_df=df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
        batch_size=batch_size,
        device=device,
    )

    output_df.to_parquet(output_path)

    mlflow.end_run()
    logger.info("==== end inference process ====")


if __name__ == "__main__":
    main()
