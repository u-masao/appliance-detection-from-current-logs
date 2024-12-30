import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # init logger
        logger = logging.getLogger(__name__)

        # logging
        logger.debug("PositionalEncoding.__init__(): %s", d_model)
        pe = torch.zeros(max_len, d_model)

        logger.debug(f"{pe.size()=}")
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        logger.debug(f"{self.pe.size()=}")  # B, Seq, E
        assert self.pe.size()[0] == 1
        assert self.pe.size()[1] == max_len
        assert self.pe.size()[2] == d_model

    def forward(self, x):  # B, Seq, E
        logger = logging.getLogger(__name__)
        logger.debug(f"PositionalEncoding.forward(): {x.size()=}")
        pe = self.pe[:, : x.size(1), :]  # 1, Seq, E
        logger.debug(f"PositionalEncoding.{pe.size()=}")
        x = x + pe
        logger.debug(f"PositionalEncoding.{x.size()=}")
        return x


class TimeSeriesModel(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        # input_sequence_length: int,
        # input_dim: int,
        # output_sequence_length: int,
        # output_dim: int,
        # embed_dim: int,
        # nhead: int = 8,
        # num_encoder_layers: int = 3,
        # num_decoder_layers: int = 3,
        # ff_dim: int = 32,
        # dropout=0.1,
    ):
        super(TimeSeriesModel, self).__init__()

        # fields
        self.config = config

        # layers
        self.src_input_projection = nn.Linear(
            config.input_dim, config.embed_dim
        )
        self.tgt_input_projection = nn.Linear(
            config.output_dim, config.embed_dim
        )
        self.positional_encoding = PositionalEncoding(config.embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.embed_dim,
                nhead=config.nhead,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=config.embed_dim,
                nhead=config.nhead,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_decoder_layers,
        )
        self.src_output_projection = nn.Linear(
            config.embed_dim, config.output_dim
        )
        self.tgt_output_projection = nn.Linear(
            config.embed_dim, config.output_dim
        )

    def forward(self, src, tgt):  # (B, inSeq, inF), (B, outSeq, outF)
        logger = logging.getLogger(__name__)
        logger.debug(f"TimeSeriesModel.forward(), src {src.size()=}")
        batch_size = src.size()[0]

        # src check
        assert src.size()[1] == self.config.input_sequence_length
        assert src.size()[2] == self.config.input_dim

        # tgt check
        if tgt is not None:
            logger.debug(f"TimeSeriesModel.forward(), tgt {src.size()=}")
            assert tgt.size()[0] == batch_size
            assert tgt.size()[1] == self.config.output_sequence_length
            assert tgt.size()[2] == self.config.output_dim

        # input projection
        src = self.src_input_projection(src)  # (B, inSec, E)
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.config.input_sequence_length
        assert src.size()[2] == self.config.embed_dim

        # positional_encoding
        src = self.positional_encoding(src)  # (B, inSec, E)
        logger.debug(f"TimeSeriesModel.forward(), PE {src.size()=}")
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.config.input_sequence_length
        assert src.size()[2] == self.config.embed_dim

        # transformer
        memory = self.encoder(src)  # B, inSec, E
        logger.debug(f"TimeSeriesModel.forward(), encoder {memory.size()=}")
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.config.input_sequence_length
        assert src.size()[2] == self.config.embed_dim

        # target processing
        tgt = self.tgt_input_projection(tgt)  # B, outSec, E
        tgt = self.positional_encoding(tgt)  # B, outSec, E
        tgt = self.decoder(tgt, memory)  # (B, outSec, E), (B, inSec, E)
        tgt = self.tgt_output_projection(tgt)  # (B, outSec, outF)
        tgt = F.relu(tgt)  # value >= 0
        logger.debug(f"TimeSeriesModel.forward(), decoder {tgt.size()=}")

        embed = memory.mean(dim=1)
        logger.debug(f"TimeSeriesModel.forward(), embed {embed.size()=}")
        assert embed.size()[0] == batch_size
        assert embed.size()[1] == self.config.embed_dim

        return tgt, embed


def create_model(config: ModelConfig):
    if config.embed_dim % config.nhead != 0:
        raise ValueError("input_dim must be divisible by nhead")

    model = TimeSeriesModel(config)
    return model


def save_model(model, path, model_config: ModelConfig):
    """Save the model and its configuration to the specified path."""
    if model_config is None:
        raise ValueError("save_model() では ModelConfig をしていしてください")

    torch.save(
        {"state_dict": model.state_dict(), "config": model_config}, path
    )


def load_model(path):
    """Load a TimeSeriesModel and its configuration from the specified path."""
    checkpoint = torch.load(path)
    model_config = checkpoint["config"]
    model = create_model(model_config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config
