import logging
import random
from pathlib import Path

import click
import mlflow
import numpy as np
import optuna
import optuna.storages
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.config import DataConfig, ModelConfig, TrainingConfig
from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import create_model, save_model


def load_and_prepare_data(
    train_path,
    val_path,
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
):
    """
    Train and evaluate the model, saving checkpoints at specified intervals.
    """
    train_df = load_data(train_path, fraction=data_config.fraction)
    val_df = load_data(val_path, fraction=data_config.fraction)
    train_dataset = TimeSeriesDataset(
        train_df,
        input_sequence_length=model_config.input_sequence_length,
        output_sequence_length=model_config.output_sequence_length,
        target_columns=training_config.target_columns,
    )
    val_dataset = TimeSeriesDataset(
        val_df,
        input_sequence_length=model_config.input_sequence_length,
        output_sequence_length=model_config.output_sequence_length,
        target_columns=training_config.target_columns,
    )
    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        num_workers=data_config.num_workers,
        shuffle=False,
        generator=generator,
    )
    return train_loader, val_loader, train_df.shape[1]


def setup_device(force_cpu):
    if force_cpu:
        return torch.device("cpu")
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_and_configure_model(
    trial, model_config: ModelConfig, training_config: TrainingConfig
):
    # Set random seeds for reproducibility
    random.seed(training_config.seed)
    np.random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    if not training_config.force_cpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_config.seed)

    logger = logging.getLogger(__name__)
    model_config.nhead = trial.suggest_int("nhead", 2, 12, step=2)
    model_config.embed_dim = trial.suggest_int(
        "embed_dim",
        model_config.nhead,
        model_config.nhead * 20,
        step=model_config.nhead,
    )
    # model_config.nhead=4
    # model_config.embed_dim=24
    logger.info(f"Trial {trial.number} params: {model_config.nhead=}")
    logger.info(f"Trial {trial.number} params: {model_config.embed_dim=}")
    # lr = trial.suggest_float("lr", 1e-4, 7.0e-4, log=True)
    # lr = 0.00010234736295408926
    # lr = 0.000514804885292421
    lr = training_config.lr
    logger.info(f"Trial {trial.number} params: {lr=}")
    mlflow.log_params({"lr": lr})

    model = create_model(model_config).to(training_config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def train_and_evaluate_model(
    trial,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    logger,
    model_config,
    training_config,
):
    min_train_loss = float("inf")
    min_val_loss = float("inf")

    device = training_config.device
    num_epochs = training_config.num_epochs
    checkpoint_interval = training_config.checkpoint_interval

    for epoch in range(num_epochs):
        # train
        model.train()
        pbar = tqdm(
            train_loader,
            desc=f"Trial {trial.number}, Epoch {epoch} [Train]",
            leave=True,
        )
        train_loss = 0
        for i, (x, y) in enumerate(pbar):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            output, embed = model(x, y)
            loss = criterion(output.view_as(y), y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            avg_train_loss = train_loss / (i + 1)
            pbar.set_postfix({"avg. loss": avg_train_loss})

        min_train_loss = min(min_train_loss, avg_train_loss)
        mlflow.log_metric("avg_loss.train", avg_train_loss, step=epoch)
        mlflow.log_metric("loss.train", train_loss, step=epoch)
        mlflow.log_metric("min_loss.train", min_train_loss, step=epoch)

        # Save checkpoint
        if (epoch) % checkpoint_interval == 0:
            checkpoint_dir = Path("models/checkpoint/")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"epoch_{epoch:0>4}.pth"
            save_model(model, checkpoint_path, model_config=model_config)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        model.eval()
        val_loss = 0
        val_iterations = 0
        with torch.no_grad():
            pbar = tqdm(
                val_loader,
                desc=f"Trial {trial.number}, Epoch {epoch} [Validation]",
                leave=True,
            )
            for i, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                output, embed = model(x, y)
                loss = criterion(output.view_as(y), y).item()
                val_loss += loss
                val_iterations += 1
                pbar.set_postfix({"loss": loss})
                avg_val_loss = val_loss / (i + 1)
                pbar.set_postfix({"avg. loss": avg_val_loss})

        min_val_loss = min(min_val_loss, avg_val_loss)
        mlflow.log_metric("avg_loss.val", avg_val_loss, step=epoch)
        mlflow.log_metric("loss.val", val_loss, step=epoch)
        mlflow.log_metric("min_loss.val", min_val_loss, step=epoch)

    return val_loss


def log_params_to_mlflow(config, prefix=""):
    mlflow.log_params(
        {f"{prefix}.{k}": v for k, v in config.model_dump().items()}
    )


def log_config_and_model(data_config, model_config, training_config, model):
    for prefix, config in [
        ["data", data_config],
        ["model", model_config],
        ["training", training_config],
    ]:
        log_params_to_mlflow(config, prefix)

    total_params = sum(
        [p.numel() for p in model.parameters() if p.requires_grad]
    )
    mlflow.log_param("model.total_params", total_params)


best_val_loss = float("inf")


def objective(
    trial,
    train_path,
    val_path,
    model_output_path,
    original_data_config: DataConfig,
    original_model_config: ModelConfig,
    original_training_config: TrainingConfig,
):
    global best_val_loss

    logger = logging.getLogger(__name__)
    mlflow.start_run(run_name=f"trial_{trial.number}", nested=True)

    # copy config for multi threadding
    data_config = original_data_config.copy(deep=True)
    model_config = original_model_config.copy(deep=True)
    training_config = original_training_config.copy(deep=True)

    # load data and make loader
    train_loader, val_loader, num_columns = load_and_prepare_data(
        train_path, val_path, data_config, model_config, training_config
    )

    # update num_columns
    data_config.num_columns = num_columns
    model_config.input_dim = data_config.num_columns - 1

    # device
    training_config.device = setup_device(training_config.force_cpu)
    logger.info(f"Using device: {training_config.device}")

    model, optimizer, criterion = create_and_configure_model(
        trial,
        model_config,
        training_config,
    )

    log_config_and_model(data_config, model_config, training_config, model)

    val_loss = train_and_evaluate_model(
        trial,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        logger,
        model_config,
        training_config,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model(model, model_output_path, model_config=model_config)
        logger.info(f"Best model saved with validation loss: {best_val_loss}")

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
    "--input_sequence_length",
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
    help="Hidden dimension for the model.",
)
@click.option(
    "--nhead",
    type=int,
    default=8,
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
    "--lr",
    type=float,
    default=0.000514804885292421,
    help="Learning ratio",
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
    "--output_sequence_length",
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
@click.option(
    "--num_workers", type=int, default=4, help="Number of dataloader workers"
)
@click.option(
    "--checkpoint_interval",
    type=int,
    default=5,
    help="Interval (in epochs) to save model checkpoints.",
)
def main(
    batch_size,
    input_sequence_length,
    embed_dim,
    nhead,
    lr,
    num_layers,
    output_sequence_length,
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
    num_workers,
    checkpoint_interval,
):
    """
    Train the transformer model with the specified parameters.
    """
    # init log
    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    mlflow.set_experiment("train_model")
    mlflow.start_run(run_name=mlflow_run_name)

    # log input args
    cli_args = click.get_current_context().params
    logger.info(f"args: {cli_args}")
    mlflow.log_params({f"args.{k}": v for k, v in cli_args.items()})

    data_config = DataConfig(
        fraction=data_fraction,
        num_workers=num_workers,
        num_columns=0,  # Placeholder, will be set in objective
    )

    training_config = TrainingConfig(
        batch_size=batch_size,
        num_epochs=num_epochs,
        checkpoint_interval=checkpoint_interval,
        lr=lr,
        force_cpu=force_cpu,
        seed=seed,
    )

    model_config = ModelConfig(
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        output_dim=len(training_config.target_columns),
        embed_dim=embed_dim,
        nhead=nhead,
    )

    storage = optuna.storages.RDBStorage(url=training_config.optuna_db_url)
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
            data_config,
            model_config,
            training_config,
        ),
        n_trials=n_trials,
        n_jobs=-1,  # Use all available cores
    )

    logger.info(f"Best trial: {study.best_trial}")

    mlflow.end_run()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = logging.DEBUG
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format=log_fmt)
    main()
