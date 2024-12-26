import torch
import torch.nn as nn


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(
        self, input_dim, embed_dim, num_heads, num_layers, output_dim
    ):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(
            embed_dim, num_heads, num_layers, batch_first=True
        )
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src).unsqueeze(1)  # Add batch dimension
        output = self.transformer(src, src)
        return self.fc_out(output.squeeze(1))  # Remove batch dimension


def create_model(input_dim, embed_dim, num_heads, num_layers, output_dim):
    """Create and initialize a TransformerModel with the specified parameters."""
    model = TransformerModel(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        output_dim=output_dim,
    )
    return model


def save_model(model, path, model_config=None):
    """Save the model and its configuration to the specified path."""
    torch.save(
        {"state_dict": model.state_dict(), "config": model_config}, path
    )


def load_model(path):
    """Load a TransformerModel and its configuration from the specified path."""
    checkpoint = torch.load(path)
    model_config = checkpoint["config"]
    model = TransformerModel(
        input_dim=model_config["input_dim"],
        embed_dim=model_config["embed_dim"],
        num_heads=model_config["num_heads"],
        num_layers=model_config["num_layers"],
        output_dim=model_config["output_dim"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model, model_config
