import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        print("PositionalEncoding.__init__(): d_model", d_model)
        pe = torch.zeros(max_len, d_model)
        print("PositionalEncoding.__init__(): pe.size()", pe.size())
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        print("self.pe.size():", self.pe.size())

    def forward(self, x):
        print("x.size():", x.size())
        print("x.size():", x.size())
        pe = self.pe[: x.size(0), :]
        print("pe.size():", pe.size())
        print("pe:", pe)
        return x + pe


class TimeSeriesModel(nn.Module):
    def __init__(self, input_sequence_length, input_dim, output_sequence_length, output_dim, hidden_dim: int = 1024):
        super(TimeSeriesModel, self).__init__()
        self.input_length = input_sequence_length
        self.output_length = output_sequence_length
        self.positional_encoding = PositionalEncoding(input_sequence_length)
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation="relu",
        )
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        print(f"TimeSeriesModel.forward(),input x.size():", x.size())
        # Reshape x to (input_length, batch_size, input_dim)
        batch_size, sequence_length, _ = x.size()
        x = x.permute(1, 0, 2)  # (input_length, batch_size, input_dim)
        x = self.positional_encoding(x)
        print(f"TimeSeriesModel.forward(),pe x.size():", x.size())
        x = self.transformer(x, x)
        print(f"TimeSeriesModel.forward(),transformer x.size():", x.size())
        x = self.fc_out(x)
        print(f"TimeSeriesModel.forward(),fc_out x.size():", x.size())
        # Reshape back to (batch_size, output_length, output_dim)
        x = x.permute(1, 0, 2)
        x = x.contiguous().view(batch_size, self.output_length, -1)
        return F.relu(x)


def create_model(input_dim, output_dim, hidden_dim):
    """Create and initialize a TimeSeriesModel with the specified parameters."""
    model = TimeSeriesModel(
        input_dim=input_dim,
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
        input_dim=model_config["input_dim"],
        output_dim=model_config["output_dim"],
        hidden_dim=model_config["hidden_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config
