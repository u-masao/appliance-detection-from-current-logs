import logging

import click
import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.models.model import TransformerModel


def load_data(file_path, fraction=1.0):
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Data types:\n{df.dtypes}")
    if fraction < 1.0:
        df = df.iloc[: int(len(df) * fraction)]
        logger.info(
            f"Data reduced to {len(df)} samples for development (sequentially)"
        )
    logger.info("Data loaded successfully")
    return df


# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length, output_length, target_columns):
        self.target_columns = target_columns
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length

    def __getitem__(self, idx):
        x = (
            self.data.iloc[idx : idx + self.input_length]
            .to_numpy(dtype="float32")
            .flatten()
        )
        y = (
            self.data.iloc[
                idx
                + self.input_length : idx
                + self.input_length
                + self.output_length
            ][self.target_columns]
            .to_numpy(dtype="float32")
            .flatten()
        )

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# Objective function for Optuna
def objective(
    trial,
    train_path,
    val_path,
    test_path,
    model_output_path,
    fraction,
    num_epochs,
    study,
    input_length,
    batch_size,
    output_length,
    train_ratio,
    val_ratio,
    force_cpu,
):
    logger = logging.getLogger(__name__)
    # Define target columns for prediction
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
    # Load data to determine the number of columns
    df = pd.read_parquet(input_path)
    num_columns = df.shape[1]
    num_heads = trial.suggest_int("num_heads", 1, 4)
    embed_dim = trial.suggest_int(
        "embed_dim", num_heads * 4, num_heads * 16, step=num_heads
    )
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    logger.info(f"params: {num_heads=}, {embed_dim=}, {num_layers=}, {lr=}")

    # Set device
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = TransformerModel(
        input_dim=input_length * num_columns,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_length * len(target_columns),
    ).to(device)
    logger.info(f"Model transferred to device: {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data
    df = load_data(input_path, fraction)
    # Load pre-split data
    train_df = pd.read_parquet(f"{input_path}_train.parquet")
    val_df = pd.read_parquet(f"{input_path}_val.parquet")
    test_df = pd.read_parquet(f"{input_path}_test.parquet")
    train_dataset = TimeSeriesDataset(
        train_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
    )
    val_dataset = TimeSeriesDataset(
        val_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1} started")
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
        for x, y in pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logger.debug(f"Data batch transferred to device: {device}")
            output = model(x)
            loss = criterion(output.view_as(y), y)
            mlflow.log_metric("train_loss", loss.item())
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=False
            )
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view_as(y), y).item()
                val_loss += loss
                pbar.set_postfix({"loss": loss})

        mlflow.log_metric("val_loss", val_loss)
    mlflow.end_run()
    logger.info("Training completed")

    logger.info(f"Final validation loss: {val_loss}")
    logger.info("==== end process ====")
    return val_loss


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("val_path", type=click.Path(exists=True))
@click.argument("test_path", type=click.Path(exists=True))
@click.argument("model_output_path", type=click.Path())
@click.option(
    "--input_length",
    type=int,
    default=60 * 3,
    help="Input length for the time series data.",
)
@click.option(
    "--mlflow_run_name",
    type=str,
    default="train_transformer",
    help="Name of the MLflow run.",
)
@click.option(
    "--data_fraction",
    type=float,
    default=1.0,
    help="Fraction of data to load for development.",
)
@click.option(
    "--num_epochs",
    type=int,
    default=10,
    help="Number of training epochs.",
)
@click.option(
    "--n_trials",
    type=int,
    default=50,
    help="Number of Optuna trials.",
)
@click.option(
    "--embed_dim",
    type=int,
    default=None,
    help="Embedding dimension for the transformer model.",
)
@click.option(
    "--num_heads",
    type=int,
    default=None,
    help="Number of attention heads for the transformer model.",
)
@click.option(
    "--num_layers",
    type=int,
    default=None,
    help="Number of layers for the transformer model.",
)
@click.option(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for the DataLoader.",
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
    "--output_length",
    type=int,
    default=5,
    help="Output length for the time series data.",
)
@click.option(
    "--force_cpu",
    is_flag=True,
    default=False,
    help="Force the use of CPU even if a GPU is available.",
)
def main(
    batch_size,
    input_length,
    embed_dim,
    num_heads,
    num_layers,
    output_length,
    input_path,
    output_path,
    mlflow_run_name,
    data_fraction,
    num_epochs,
    n_trials,
    train_ratio,
    val_ratio,
    force_cpu,
):
    """
    Train the transformer model with the specified parameters.

    :param input_path: Path to the input data file.
    :param output_path: Path to save the output model.
    :param mlflow_run_name: Name of the MLflow run.
    :param data_fraction: Fraction of data to use for training.
    :param num_epochs: Number of training epochs.
    """
    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    logger.info(f"Train path: {train_path}")
    logger.info(f"Validation path: {val_path}")
    logger.info(f"Test path: {test_path}")
    logger.info(f"Model output path: {model_output_path}")

    mlflow.set_experiment("train_transformer")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params({
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "model_output_path": model_output_path
    })

    df = pd.read_parquet(train_path)
    num_columns = df.shape[1]
    # Define target columns for prediction
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(
            trial,
            input_path,
            output_path,
            data_fraction,
            num_epochs,
            study,
            input_length,
            batch_size,
            output_length,
            train_ratio,
            val_ratio,
            force_cpu,
        ),
        n_trials=n_trials,
    )
    logger.info(f"Best trial: {study.best_trial}")
    # Define the model with the best parameters
    best_trial = study.best_trial
    # Use CLI options if provided, otherwise use best trial parameters
    embed_dim = (
        embed_dim if embed_dim is not None else best_trial.params["embed_dim"]
    )
    num_heads = (
        num_heads if num_heads is not None else best_trial.params["num_heads"]
    )
    num_layers = (
        num_layers
        if num_layers is not None
        else best_trial.params["num_layers"]
    )
    model = TransformerModel(
        input_dim=input_length * num_columns,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_length * len(target_columns),
    )
    # Save the best model
    torch.save(model.state_dict(), output_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
