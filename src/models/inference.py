import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

def run_inference(model, test_df, input_length, output_length, target_columns, batch_size, device):
    # Create test dataset and loader
    test_dataset = TimeSeriesDataset(
        test_df,
        input_length=input_length,
        output_length=output_length,
        target_columns=target_columns,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference [Test]"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())

    # Convert lists to arrays
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actuals[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted on Test Set")
    plt.legend()
    plt.show()
