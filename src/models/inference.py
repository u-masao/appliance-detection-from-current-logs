import logging
import random

import click
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dataset import TimeSeriesDataset
from src.models.model import load_model
from src.models.train_model import load_data


def run_inference(
    model,
    test_df,
    input_length,
    output_length,
    target_columns,
    batch_size,
    device,
    seed,
):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    logger = logging.getLogger(__name__)
    # Create test dataset and loader
    test_dataset = TimeSeriesDataset(
        test_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
    )
    generator = torch.Generator().manual_seed(seed)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, generator=generator
    )

    model.eval()
    trains = []
    predictions = []
    actuals = []
    embeds = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference"):
            x, y = x.to(device), y.to(device)
            output, embed = model(x, y)
            trains.append(x.cpu().numpy().reshape(len(x), -1))
            predictions.append(output.cpu().numpy().reshape(len(x), -1))
            actuals.append(y.cpu().numpy().reshape(len(x), -1))
            embeds.append(embed.cpu().numpy().reshape(len(x), -1))

    # Convert lists to arrays
    trains = np.concatenate(trains, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    embeds = np.concatenate(embeds, axis=0)

    logger.info(
        f"shapes: {trains.shape=}, {predictions.shape=}, "
        f"{actuals.shape=}, {embeds.shape=}"
    )

    train_df = pd.DataFrame(trains).add_prefix("train_")
    pred_df = pd.DataFrame(predictions).add_prefix("pred_")
    actual_df = pd.DataFrame(actuals).add_prefix("actual_")
    embed_df = pd.DataFrame(embeds).add_prefix("embed_")
    concat_df = pd.concat([train_df, pred_df, actual_df, embed_df], axis=1)

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
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
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
    seed,
):
    logger = logging.getLogger(__name__)
    logger.info("==== start inference process ====")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Input path: {input_path}")

    mlflow.set_experiment("inference")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params({"model_path": model_path, "input_path": input_path})
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]

    # Load model and configuration
    model, model_config = load_model(model_path)
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
        seed=seed,
    )

    output_df.to_parquet(output_path)

    mlflow.end_run()
    logger.info("==== end inference process ====")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
