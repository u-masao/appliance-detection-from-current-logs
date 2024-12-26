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

def save_model(model, path):
    """Save the model to the specified path."""
    torch.save(model.state_dict(), path)

def load_model(path, model_class, *args, **kwargs):
    """Load a model from the specified path."""
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model
