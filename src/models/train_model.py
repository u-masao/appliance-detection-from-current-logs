import logging
import random

import click
import mlflow
import numpy as np
import optuna
import optuna.storages
import pandas as pd
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import create_model, save_model


def load_and_prepare_data(
    train_path,
    val_path,
    fraction,
    input_length,
    output_length,
    batch_size,
    target_columns,
    seed,
):
    train_df = load_data(train_path, fraction=fraction)
    val_df = load_data(val_path, fraction=fraction)
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
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        generator=generator,
    )
    return train_loader, val_loader, train_df.shape[1]


def setup_device(force_cpu):
    if force_cpu:
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
def create_and_configure_model(
    trial,
    input_length,
    num_columns,
    output_length,
    target_columns,
    device,
    force_cpu,
    seed,
):
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not force_cpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = logging.getLogger(__name__)
    lr = trial.suggest_float("lr", 5e-3, 5e-2, log=True)
    logger.info(f"params: {lr=}")
    mlflow.log_params({"lr": lr})

    model = create_model(
        input_dim=input_length * (num_columns - 1),
        output_dim=output_length * len(target_columns),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def train_and_evaluate_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    logger,
):
    min_train_loss = float('inf')
    min_val_loss = float('inf')

    for epoch in range(num_epochs):
        # train
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=True)
        train_loss = 0
        for i, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.view_as(y), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            avg_train_loss = train_loss / (i + 1)
            pbar.set_postfix({"avg. loss": avg_train_loss})

        min_train_loss = min(min_train_loss, avg_train_loss)
        mlflow.log_metric("avg_loss.train", avg_train_loss, step=epoch + 1)
        mlflow.log_metric("loss.train", train_loss, step=epoch + 1)
        mlflow.log_metric("min_loss.train", min_train_loss, step=epoch + 1)

        # validation
        model.eval()
        val_loss = 0
        val_iterations = 0
        with torch.no_grad():
            pbar = tqdm(
                val_loader, desc=f"Epoch {epoch+1} [Validation]", leave=True
            )
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.view_as(y), y).item()
                val_loss += loss
                val_iterations += 1
                pbar.set_postfix({"loss": loss})
                avg_val_loss = val_loss / (i + 1)
                pbar.set_postfix({"avg. loss": avg_val_loss})

        min_val_loss = min(min_val_loss, avg_val_loss)
        mlflow.log_metric("avg_loss.val", avg_val_loss, step=epoch + 1)
        mlflow.log_metric("loss.val", val_loss, step=epoch + 1)
        mlflow.log_metric("min_loss.val", min_val_loss, step=epoch + 1)

    return val_loss


def objective(
    trial,
    train_path,
    val_path,
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
    seed,
):
    logger = logging.getLogger(__name__)
    mlflow.start_run()
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
    train_loader, val_loader, num_columns = load_and_prepare_data(
        train_path,
        val_path,
        fraction,
        input_length,
        output_length,
        batch_size,
        target_columns,
        seed=seed,
    )
    device = setup_device(force_cpu)
    logger.info(f"Using device: {device}")
    model, optimizer, criterion = create_and_configure_model(
        trial,
        input_length,
        num_columns,
        output_length,
        target_columns,
        device,
        force_cpu,
        seed,
    )
    val_loss = train_and_evaluate_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        num_epochs,
        logger,
    )
    mlflow.end_run()
    logger.info("Training completed")
    logger.info(f"Final validation loss: {val_loss}")
    logger.info("==== end process ====")
    return val_loss


@click.command()
@click.argument("train_path", type=click.Path(exists=True))
@click.argument("val_path", type=click.Path(exists=True))
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
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
def main(
    batch_size,
    input_length,
    embed_dim,
    num_heads,
    num_layers,
    output_length,
    train_path,
    val_path,
    model_output_path,
    mlflow_run_name,
    data_fraction,
    num_epochs,
    n_trials,
    train_ratio,
    val_ratio,
    force_cpu,
    seed,
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
    logger.info(f"Model output path: {model_output_path}")

    mlflow.set_experiment("train_model")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params(
        {
            "train_path": train_path,
            "val_path": val_path,
            "model_output_path": model_output_path,
        }
    )

    # Define target columns for prediction
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
    storage = optuna.storages.RDBStorage(
        url="sqlite:///./data/interim/optuna_study.db"
    )
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        storage=storage, direction="minimize", sampler=sampler
    )

    study.optimize(
        lambda trial: objective(
            trial,
            train_path,
            val_path,
            model_output_path,
            data_fraction,
            num_epochs,
            study,
            input_length,
            batch_size,
            output_length,
            train_ratio,
            val_ratio,
            force_cpu,
            seed,
        ),
        n_trials=n_trials,
        n_jobs=-1,  # Use all available cores
    )
    logger.info(f"Best trial: {study.best_trial}")
    # Define the model with the best parameters
    best_trial = study.best_trial
    # Use CLI options if provided, otherwise use best trial parameters

    # Load data to determine the number of columns
    train_df = load_data(train_path, fraction=data_fraction)
    num_columns = train_df.shape[1]
    model_config = {
        "input_dim": input_length * (num_columns - 1),
        "output_dim": output_length * len(target_columns),
    }
    model = create_model(**model_config)
    # Output model architecture
    logger.debug("Model architecture:")
    logger.debug(model)
    save_model(model, model_output_path, model_config=model_config)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
