import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import optuna
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    df = pd.read_parquet(file_path)
    df = df[df['Gap'] == False]  # Filter out non-continuous data
    return df

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_length=30, output_length=5):
        self.data = data
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.data) - self.input_length - self.output_length

    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.input_length].values
        y = self.data.iloc[idx+self.input_length:idx+self.input_length+self.output_length][['watt_black', 'watt_red', 'watt_kitchen', 'watt_living']].values
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer = nn.Transformer(embed_dim, num_heads, num_layers)
        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        return self.fc_out(output[-1])

# Objective function for Optuna
def objective(trial):
    # Hyperparameters
    embed_dim = trial.suggest_int('embed_dim', 16, 64)
    num_heads = trial.suggest_int('num_heads', 1, 4)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)

    # Model
    model = TransformerModel(input_dim=5, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, output_dim=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data
    df = load_data('data/interim/features.parquet')
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    train_dataset = TimeSeriesDataset(train_df)
    val_dataset = TimeSeriesDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Training loop
    for epoch in range(10):  # Simple loop for demonstration
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                output = model(x)
                val_loss += criterion(output, y).item()

    return val_loss

# Run Optuna
if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print("Best trial:", study.best_trial)
