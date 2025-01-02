import torch

from src.models.config import ModelConfig
from src.models.model import PositionalEncoding, TimeSeriesModel


def test_positional_encoding():
    d_model = 16
    max_len = 10
    batch_size = 5
    seq_length = 10

    positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

    # Create a dummy input (batch_size, seq_length, d_model)
    dummy_input = torch.zeros(batch_size, seq_length, d_model)

    # Apply positional encoding
    encoded_output = positional_encoding(dummy_input)

    # Check if the output shape is as expected
    assert encoded_output.shape == dummy_input.shape, (
        f"Expected encoded output shape {dummy_input.shape}, "
        f"but got {encoded_output.shape}"
    )

    # Check if the positional encoding (output should not be all zeros)
    assert not torch.all(
        encoded_output == 0
    ), "Positional encoding not applied"
    input_sequence_length = 10
    input_dim = 3
    output_sequence_length = 4
    output_dim = 2
    embed_dim = 16
    batch_size = 5
    model_config = ModelConfig(
        input_sequence_length=input_sequence_length,
        input_dim=input_dim,
        output_sequence_length=output_sequence_length,
        output_dim=output_dim,
        embed_dim=embed_dim,
    )

    model = TimeSeriesModel(model_config)
    model.eval()  # Set model to evaluation mode

    # Create a dummy input tensor
    # (batch_size, input_sequence_length, input_dim)
    dummy_input = torch.randn(batch_size, input_sequence_length, input_dim)

    # Create a dummy target tensor
    # (batch_size, output_sequence_length, output_dim)
    dummy_target = torch.randn(batch_size, output_sequence_length, output_dim)
    # Perform a forward pass and ensure tgt is not None
    try:
        tgt, embed = model(dummy_input, dummy_target)
    except Exception as e:
        assert False, f"Model forward pass raised an exception: {e}"
    assert tgt is not None, "Expected tgt to be not None"

    # Check if the output shapes are as expected
    assert tgt.shape == (
        batch_size,
        output_sequence_length,
        output_dim,
    ), (
        "Expected tgt shape "
        f"{(batch_size, output_sequence_length, output_dim)}, "
        f"but got {tgt.shape}"
    )
    assert embed is not None, "Expected embed to be not None"
    assert embed.shape == (
        batch_size,
        embed_dim,
    ), (
        "Expected embed shape "
        f"{(batch_size, embed_dim)}, but got {embed.shape}"
    )
