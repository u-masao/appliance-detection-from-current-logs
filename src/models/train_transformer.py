import logging

import click
import mlflow
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, data, input_length=60 * 3, output_length=5):
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
            ][["watt_black", "watt_red", "watt_kitchen", "watt_living"]]
            .to_numpy(dtype="float32")
            .flatten()
        )
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, embed_dim, num_heads, num_layers, output_dim
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        return self.fc_out(output[-1])


# Objective function for Optuna
def objective(trial, input_path, output_path, fraction):
    logger = logging.getLogger(__name__)
    # Hyperparameters
    embed_dim = trial.suggest_int(
        "embed_dim", num_heads * 4, num_heads * 16, step=num_heads
    )
    num_heads = trial.suggest_int("num_heads", 1, 4)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)

    logger.info(f"params: {num_heads=}, {embed_dim=}, {num_layers=}, {lr=}")

    # Model
    model = TransformerModel(
        input_dim=60 * 3 * 13,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=5 * 4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data
    df = load_data(input_path, fraction)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(10):
        logger.info(f"Epoch {epoch+1} started")
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view_as(y), y)
            mlflow.log_metric("train_loss", loss.item())
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                val_loss += criterion(output.view_as(y), y).item()

        mlflow.log_metric("val_loss", val_loss)
    mlflow.end_run()
    logger.info("Training completed")
    logger.info(f"Final validation loss: {val_loss}")
    logger.info("==== end process ====")
    return val_loss


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
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
def main(input_path, output_path, mlflow_run_name, data_fraction):
    logger = logging.getLogger(__name__)
    logger.info("==== start process ====")
    logger.info(f"Input path: {input_path}")
    logger.info(f"Output path: {output_path}")

    mlflow.set_experiment("train_transformer")
    mlflow.start_run(run_name=mlflow_run_name)
    mlflow.log_params({"input_path": input_path, "output_path": output_path})

    # Run Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, input_path, output_path, data_fraction),
        n_trials=50,
    )
    logger.info(f"Best trial: {study.best_trial}")
    # Save the best model
    torch.save(study.best_trial.value, output_path)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
