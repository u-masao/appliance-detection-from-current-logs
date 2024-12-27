import pandas as pd
import torch
from src.models.dataset import TimeSeriesDataset

def test_timeseriesdataset():
    # サンプルデータを作成
    data = {
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "gap": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "watt_black": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "watt_red": [15, 25, 35, 45, 55, 65, 75, 85, 95, 105],
        "watt_kitchen": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
        "watt_living": [25, 35, 45, 55, 65, 75, 85, 95, 105, 115],
    }
    df = pd.DataFrame(data)

    # パラメータを設定
    input_length = 3
    output_length = 2
    target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]

    # データセットを作成
    dataset = TimeSeriesDataset(df, input_length, output_length, target_columns)

    # データセットの長さをテスト
    assert len(dataset) == 6  # 有効なインデックスは0から5まで

    # データセットのアイテムをテスト
    x, y = dataset[0]
    assert torch.equal(x, torch.tensor([[1, 2], [2, 3], [3, 4]], dtype=torch.float32))
    assert torch.equal(y, torch.tensor([[40, 45, 50, 55], [50, 55, 60, 65]], dtype=torch.float32))
