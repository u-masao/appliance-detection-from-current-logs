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
        logger.info(f"{self.pe.size()=}")  # B, Seq, E
        assert self.pe.size()[0] == 1
        assert self.pe.size()[1] == max_len
        assert self.pe.size()[2] == d_model

    def forward(self, x):  # B, Seq, E
        logger = logging.getLogger(__name__)
        logger.info(f"PositionalEncoding.forward(): {x.size()=}")
        pe = self.pe[:, : x.size(1), :]  # 1, Seq, E
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
        embed_dim: int,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        ff_dim: int = 32,
        dropout=0.1,
    ):
        super(TimeSeriesModel, self).__init__()
        # init logger
        logger = logging.getLogger(__name__)

        # fields
        self.input_sequence_length = input_sequence_length
        self.input_dim = input_dim
        self.output_sequence_length = output_sequence_length
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # layers
        self.src_input_projection = nn.Linear(input_dim, embed_dim)
        self.tgt_input_projection = nn.Linear(output_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_encoder_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=nhead,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_decoder_layers,
        )
        self.src_output_projection = nn.Linear(embed_dim, output_dim)
        self.tgt_output_projection = nn.Linear(embed_dim, output_dim)

    def forward(self, src, tgt):  # (B, inSeq, inF), (B, outSeq, outF)
        logger = logging.getLogger(__name__)
        logger.info(f"TimeSeriesModel.forward(), src {src.size()=}")
        batch_size = src.size()[0]

        # src check
        assert src.size()[1] == self.input_sequence_length
        assert src.size()[2] == self.input_dim

        # tgt check
        if tgt is not None:
            logger.info(f"TimeSeriesModel.forward(), tgt {src.size()=}")
            assert tgt.size()[0] == batch_size
            assert tgt.size()[1] == self.output_sequence_length
            assert tgt.size()[2] == self.output_dim

        # input projection
        src = self.src_input_projection(src)  # (B, inSec, E)
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.input_sequence_length
        assert src.size()[2] == self.embed_dim

        # positional_encoding
        src = self.positional_encoding(src)  # (B, inSec, E)
        logger.info(f"TimeSeriesModel.forward(), PE {src.size()=}")
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.input_sequence_length
        assert src.size()[2] == self.embed_dim

        # transformer
        memory = self.encoder(src)  # B, inSec, E
        logger.info(f"TimeSeriesModel.forward(), encoder {memory.size()=}")
        assert src.size()[0] == batch_size
        assert src.size()[1] == self.input_sequence_length
        assert src.size()[2] == self.embed_dim

        # target processing
        tgt = self.tgt_input_projection(tgt)  # B, outSec, E
        tgt = self.positional_encoding(tgt)  # B, outSec, E
        tgt = self.decoder(tgt, memory)  # (B, outSec, E), (B, outSec, E)
        tgt = self.tgt_output_projection(tgt)  # (B, outSec, outF)
        tgt = F.relu(tgt)  # value >= 0
        logger.info(f"TimeSeriesModel.forward(), decoder {tgt.size()=}")

        embed = memory.mean(dim=1)
        logger.info(f"TimeSeriesModel.forward(), embed {embed.size()=}")
        assert embed.size()[0] == batch_size
        assert embed.size()[1] == self.embed_dim

        return tgt, embed


def create_model(
    input_sequence_length,
    input_dim,
    output_sequence_length,
    output_dim,
    embed_dim,
    nhead: int = 8,
):
    """Create and initialize a TimeSeriesModel with the specified parameters."""
    if embed_dim % nhead != 0:
        raise ValueError("input_dim must be divisible by nhead")

    model = TimeSeriesModel(
        input_sequence_length=input_sequence_length,
        input_dim=input_dim,
        output_sequence_length=output_sequence_length,
        output_dim=output_dim,
        embed_dim=embed_dim,
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
        embed_dim=model_config["embed_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config
