import torch

from src.models.model import TimeSeriesModel


def test_timeseries_model_forward():
    input_dim = 30  # 10 * 3 for input dimension
    output_dim = 8  # 4 * 2 for output dimension
    batch_size = 5  # Example batch size

    model = TimeSeriesModel(input_dim=input_dim, output_dim=output_dim)
    print(model)
    model.eval()  # Set model to evaluation mode

    # Create a dummy input tensor with the shape (batch_size, input_dim)
    dummy_input = torch.randn(batch_size, input_dim)

    # Perform a forward pass
    output = model(dummy_input)

    # Check if the output shape is as expected
    assert output.shape == (
        batch_size,
        output_dim,
    ), f"Expected output shape {(batch_size, output_dim)}, but got {output.shape}"
