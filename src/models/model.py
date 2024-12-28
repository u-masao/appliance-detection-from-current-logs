import math

import torch
import torch.nn as nn


class TimeSeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim: int = 1024):
        super(TimeSeriesModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))


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
