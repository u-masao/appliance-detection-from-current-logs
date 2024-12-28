import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # init logger
        logger = logging.getLogger(__name__)

        # logging
        logger.info("PositionalEncoding.__init__(): %s", d_model)
        pe = torch.zeros(max_len, d_model)

        logger.info(f"{pe.size()=}")
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        logger.info(f"{self.pe.size()=}")

    def forward(self, x):
        logger = logging.getLogger(__name__)
        logger.info(f"PositionalEncoding.{x.size()=}")
        pe = self.pe[: x.size(0), :]
        logger.info(f"PositionalEncoding.{pe.size()=}")
        return x + pe


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_sequence_length,
        input_dim,
        output_sequence_length,
        output_dim,
        hidden_dim: int = 1024,
    ):
        super(TimeSeriesModel, self).__init__()
        logger = logging.getLogger(__name__)
        self.input_length = input_sequence_length
        self.output_length = output_sequence_length
        self.positional_encoding = PositionalEncoding(input_sequence_length)
        logger.info(f"{self.positional_encoding=}")

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=hidden_dim * 8,
            dropout=0.1,
            activation="relu",
        )
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        logger = logging.getLogger(__name__)
        logger.info(f"TimeSeriesModel.forward(),input {x.size()=}")
        # Reshape x to (input_length, batch_size, input_dim)
        batch_size, sequence_length, _ = x.size()
        x = x.permute(1, 0, 2)  # (input_length, batch_size, input_dim)
        logger.info(f"TimeSeriesModel.forward(),permute {x.size()=}")
        x = self.positional_encoding(x)
        logger.info(f"TimeSeriesModel.forward(),pe {x.size()=}")
        x = self.transformer(x, x)
        logger.info(f"TimeSeriesModel.forward(),transformer {x.size()=}")
        x = self.fc_out(x)
        logger.info(f"TimeSeriesModel.forward(),fc_out {x.size()=}")
        x = x.permute(1, 0, 2)
        logger.info(f"TimeSeriesModel.forward(),permute {x.size()=}")
        x = x.contiguous().view(batch_size, self.output_length, -1)
        logger.info(f"TimeSeriesModel.forward(),contiguous {x.size()=}")
        return F.relu(x)


def create_model(
    input_sequence_length,
    input_dim,
    output_sequence_length,
    output_dim,
    hidden_dim,
    nhead: int = 8,
):
    """Create and initialize a TimeSeriesModel with the specified parameters."""
    if hidden_dim % nhead != 0:
        raise ValueError("input_dim must be divisible by nhead")

    model = TimeSeriesModel(
        input_sequence_length=input_sequence_length,
        input_dim=input_dim,
        output_sequence_length=output_sequence_length,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )
    return model


def save_model(model, path, model_config=None):
    """Save the model and its configuration to the specified path."""
    torch.save(
        {"state_dict": model.state_dict(), "config": model_config}, path
    )


def load_model(path):
    """Load a TimeSeriesModel and its configuration from the specified path."""
    checkpoint = torch.load(path)
    model_config = checkpoint["config"]
    model = create_model(
        input_sequence_length=model_config["input_sequence_length"],
        input_dim=model_config["input_dim"],
        output_sequence_length=model_config["output_sequence_length"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config
