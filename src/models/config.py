import torch
from pydantic import BaseModel


class DataConfig(BaseModel):
    num_columns: int = 0
    fraction: float = 1.0
    num_workers: int = 4


class ModelConfig(BaseModel):
    input_sequence_length: int
    input_dim: int
    output_sequence_length: int
    output_dim: int
    embed_dim: int
    nhead: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    ff_dim: int = 32
    dropout = 0.1


class TrainingConfig(BaseModel):
    batch_size: int = 32
    num_epochs: int = 10
    checkpoint_interval: int = 5
    device: torch.device = torch.device("cpu")
    force_cpu: bool = False
    seed: int = 42
    optuna_db_url: str = "sqlite:///./data/interim/optuna_study.db"
    target_columns: list = [
        "watt_black",
        "watt_red",
        "watt_kitchen",
        "watt_living",
    ]

    lr: float = 0.000514804885292421

    class Config:
        arbitrary_types_allowed = True
