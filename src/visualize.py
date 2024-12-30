import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import load_model

target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
model_filepath = "models/best_model.pth"
# model_filepath = "models/checkpoint/epoch_0012.pth"
fraction = 0.01

input_df = None
model = None
dataset = None
infered_df = None


def load_input_data(data_type):
    global input_df, model, dataset, infered_df

    if data_type not in ["train", "val", "test"]:
        gr.Warning("invalid data type")
        return
    input_filepath = f"data/interim/{data_type}.parquet"
    infered_filepath = f"data/interim/infer_{data_type}.parquet"

    # load model
    model, model_config = load_model(model_filepath)
    print(f"{model=}")
    print(f"{model_config=}")

    model.eval()
    # load data
    input_df = load_data(input_filepath, fraction=fraction)
    print(f"{input_df.columns=}")

    infered_df = load_data(infered_filepath, fraction=fraction)
    print(f"{infered_df.columns=}")

    # make dataset
    dataset = TimeSeriesDataset(
        data=input_df,
        input_sequence_length=model_config.input_sequence_length,
        output_sequence_length=model_config.output_sequence_length,
        target_columns=target_columns,
    )

    # inform data loaded
    gr.Info(f"data loaded: {data_type}")


load_input_data("train")


def take_infered_data(num_index):
    return infered_df.iloc[num_index:, :].head(10)


def create_dataframes(x, y, output):
    train_df = pd.DataFrame(
        x, columns=input_df.iloc[[0]].drop("gap", axis=1).columns
    )
    actual_df = pd.DataFrame(y, columns=target_columns)
    pred_df = pd.DataFrame(output, columns=target_columns)
    append_df = pd.concat(
        [actual_df, pred_df.add_prefix("pred_")], axis=1
    ).assign(predict=1)
    concat_df = pd.concat([train_df, append_df]).reset_index()
    return concat_df, append_df


def create_plot(concat_df, append_df):
    plt.close()
    pred_columns = [f"pred_{x}" for x in target_columns]
    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 4)
    full_ax = fig.add_subplot(gs[0, :])
    detail_axes = [
        fig.add_subplot(gs[1, x], sharey=full_ax)
        for x in range(len(target_columns))
    ]

    for column in [target_columns + ["predict"]]:
        full_ax.plot(concat_df[column], label=column, alpha=0.8)

    for index, (actual_column, pred_column) in enumerate(
        zip(target_columns, pred_columns)
    ):
        detail_axes[index].plot(
            append_df[actual_column], label=actual_column, alpha=0.8
        )
        detail_axes[index].plot(
            append_df[pred_column], label=actual_column, alpha=0.8
        )

    for ax in [full_ax] + detail_axes:
        ax.legend()
        ax.grid()
    fig.tight_layout()
    return fig


def perform_inference(data_index):
    with torch.no_grad():
        x, y = dataset[data_index]
        zero_y = torch.zeros(y.size())
        output, embed = model(x.unsqueeze(0), y.unsqueeze(0))
        output_zero, embed_zero = model(x.unsqueeze(0), zero_y.unsqueeze(0))

        concat_df, append_df = create_dataframes(x, y, output[0])
        concat_zero_df, append_zero_df = create_dataframes(
            x, y, output_zero[0]
        )
        fig = create_plot(concat_df, append_df)
        fig_zero = create_plot(concat_zero_df, append_zero_df)

    return (
        gr.Plot(value=fig),
        gr.Plot(value=fig_zero),
        gr.DataFrame(concat_df.tail(60)),
        gr.DataFrame(concat_zero_df.tail(60)),
    )


with gr.Blocks() as demo:
    gr.Markdown("# Parquet Data Viewer")

    with gr.Row():
        input_filepath = gr.Dropdown(choices=["train", "val", "test"])

    with gr.Tab("model check"):
        model_data_index = gr.Number(value=0, minimum=0, maximum=len(dataset))
        model_data_reload_button = gr.Button("reload")
        model_output_box = gr.Plot()
        model_output_box_zero = gr.Plot()
        model_output_dataframe = gr.DataFrame()
        model_output_dataframe_zero = gr.DataFrame()

    with gr.Tab("data check"):
        infered_data_index = gr.Number(
            value=0, minimum=0, maximum=len(infered_df)
        )
        infered_output_table = gr.DataFrame()

    input_filepath.select(
        load_input_data,
        inputs=[input_filepath],
        outputs=[infered_output_table],
    )
    model_data_index.change(
        fn=perform_inference,
        inputs=model_data_index,
        outputs=[
            model_output_box,
            model_output_box_zero,
            model_output_dataframe,
            model_output_dataframe_zero,
        ],
    )
    infered_data_index.change(
        fn=take_infered_data,
        inputs=infered_data_index,
        outputs=[infered_output_table],
    )
if __name__ == "__main__":
    demo.launch(share=True)
