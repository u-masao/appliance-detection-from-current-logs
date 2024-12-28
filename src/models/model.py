import math
import torch.nn.functional as F

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    def __init__(self, input_dim, output_dim, hidden_dim: int = 1024):
        super(TimeSeriesModel, self).__init__()
        self.positional_encoding = PositionalEncoding(input_dim)
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            activation='relu'
        )
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.transformer(x, x)
        x = self.fc_out(x)
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
