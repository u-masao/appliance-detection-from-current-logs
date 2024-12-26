import logging

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from model import TransformerModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_transformer import TimeSeriesDataset, load_data


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
    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference [Test]"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())

    # Convert lists to arrays
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted on Test Set")
    plt.legend()
    plt.show()


@click.command()
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--fraction",
    type=float,
    default=1.0,
    help="Fraction of data to load for testing.",
)
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
    "--input_length", type=int, required=True, help="Input sequence length."
)
@click.option(
    "--output_length", type=int, required=True, help="Output sequence length."
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
    fraction,
    train_ratio,
    val_ratio,
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

    # Load model
    model = TransformerModel(
        input_dim=len(target_columns),
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        output_dim=len(target_columns),
    )
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Load and split data
    df = load_data(input_path, fraction)
    train_size = int(len(df) * train_ratio)
    val_size = int(len(df) * val_ratio)
    test_df = df.iloc[train_size + val_size :]

    # Run inference
    run_inference(
        model=model,
        test_df=test_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
        batch_size=batch_size,
        device=device,
    )

    mlflow.end_run()
    logger.info("==== end inference process ====")


if __name__ == "__main__":
    main()
