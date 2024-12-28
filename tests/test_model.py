import torch

from src.models.model import TimeSeriesModel


def test_timeseries_model_forward():
    input_sequence_length = 10
    input_dim = 3
    output_sequence_length = 4
    output_dim = 2
    embed_dim = 16
    batch_size = 5

    model = TimeSeriesModel(
        input_sequence_length=input_sequence_length,
        input_dim=input_dim,
        output_sequence_length=output_sequence_length,
        output_dim=output_dim,
        embed_dim=embed_dim,
    )
    model.eval()  # Set model to evaluation mode

    # Create a dummy input tensor with the shape (batch_size, input_sequence_length, input_dim)
    dummy_input = torch.randn(batch_size, input_sequence_length, input_dim)

    # Perform a forward pass and catch any exceptions
    try:
        tgt, embed = model(dummy_input)
    except Exception as e:
        assert False, f"Model forward pass raised an exception: {e}"
    tgt, embed = model(dummy_input)

    # Check if the output shapes are as expected
    assert tgt.shape == (
        batch_size,
        output_sequence_length,
        output_dim,
    ), f"Expected tgt shape {(batch_size, output_sequence_length, output_dim)}, but got {tgt.shape}"
    assert embed.shape == (
        batch_size,
        embed_dim,
    ), f"Expected embed shape {(batch_size, embed_dim)}, but got {embed.shape}"
