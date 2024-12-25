import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_transformer import TimeSeriesDataset, load_data
from model import TransformerModel
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
def main():
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of data to load for testing.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Ratio of data to use for training.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Ratio of data to use for validation.")
    parser.add_argument("--input-length", type=int, required=True, help="Input sequence length.")
    parser.add_argument("--output-length", type=int, required=True, help="Output sequence length.")
    parser.add_argument("--target-columns", type=int, nargs='+', required=True, help="Indices of target columns.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu or cuda).")

    args = parser.parse_args()

    # Load model
    model = TransformerModel(input_dim=len(args.target_columns), embed_dim=64, num_heads=4, num_layers=2, output_dim=len(args.target_columns))
    model.load_state_dict(torch.load(args.model_path))
    model.to(args.device)

    # Load and split data
    df = load_data(args.input_path, args.fraction)
    train_size = int(len(df) * args.train_ratio)
    val_size = int(len(df) * args.val_ratio)
    test_df = df.iloc[train_size + val_size :]

    # Run inference
    run_inference(
        model=model,
        test_df=test_df,
        input_length=args.input_length,
        output_length=args.output_length,
        target_columns=args.target_columns,
        batch_size=args.batch_size,
        device=args.device
    )

if __name__ == "__main__":
    main()
