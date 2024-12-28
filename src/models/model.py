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
        logger.info(f"{self.pe.size()=}")  # B, Seq, H

    def forward(self, x):  # B, Seq, H
        logger = logging.getLogger(__name__)
        logger.info(f"PositionalEncoding.forward(): {x.size()=}")
        pe = self.pe[:, : x.size(1), :]  # 1, Seq, H
        logger.info(f"PositionalEncoding.{pe.size()=}")
        x = x + pe
        logger.info(f"PositionalEncoding.{x.size()=}")
        return x


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        input_sequence_length: int,
        input_dim: int,
        output_sequence_length: int,
        output_dim: int,
        hidden_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward_ratio: int = 8,
        dropout=0.1,
    ):
        super(TimeSeriesModel, self).__init__()
        logger = logging.getLogger(__name__)
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.hidden_dim = hidden_dim

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        logger.info(f"{self.positional_encoding=}")

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=hidden_dim * dim_feedforward_ratio,
            dropout=dropout,
            activation="relu",
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # B, Seq, F
        logger = logging.getLogger(__name__)
        logger.info(f"TimeSeriesModel.forward(),input {x.size()=}")
        batch_size, sequence_length, input_dim = x.size()

        # convert feature to embedding
        x = self.embedding(x)  # B, Seq, H
        assert x.size()[2] == self.hidden_dim

        # positional_encoding
        x = self.positional_encoding(x)
        logger.info(f"TimeSeriesModel.forward(), PE {x.size()=}")
        assert x.size()[0] == batch_size
        assert x.size()[1] == sequence_length
        assert x.size()[2] == self.hidden_dim

        # permute
        x = x.permute(1, 0, 2)  # Seq, B, H
        assert x.size()[0] == sequence_length
        assert x.size()[1] == batch_size
        assert x.size()[2] == self.hidden_dim

        # transformer
        x = self.transformer(x, x) # [: self.output_length])
        logger.info(f"TimeSeriesModel.forward(),transformer {x.size()=}")

        assert False
        x = self.fc_out(x)
        logger.info(f"TimeSeriesModel.forward(),fc_out {x.size()=}")
        x = x.permute(
            1, 0, 2
        )  # (batch_size, output_sequence_length, hidden_dim)
        logger.info(f"TimeSeriesModel.forward(),permute {x.size()=}")
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
