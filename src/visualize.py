import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import load_model

target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]
model_filepath = "models/best_model.pth"
fraction = 0.01

input_df = None
model = None
dataset = None


def load_input_data(input_filepath):
    global input_df, model, dataset

    # load data
    input_df = load_data(input_filepath, fraction=fraction)

    # load model
    model, model_config = load_model(model_filepath)
    model.eval()

    # make dataset
    dataset = TimeSeriesDataset(
        data=input_df,
        input_length=model_config["input_sequence_length"],
        output_length=model_config["output_sequence_length"],
        target_columns=target_columns,
    )

    # inform data loaded
    gr.Info(f"data loaded: {input_filepath}")


load_input_data("data/interim/train.parquet")


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
        output, embed = model(x.unsqueeze(0), y.unsqueeze(0))
        output = output[0]  # B, outSeq, outF -> outSeq, outF
        embed = embed[0]  # B, E -> # E

        concat_df, append_df = create_dataframes(x, y, output)
        fig = create_plot(concat_df, append_df)

    return gr.Plot(value=fig), gr.DataFrame(concat_df)


with gr.Blocks() as demo:
    gr.Markdown("# Parquet Data Viewer")

    with gr.Row():
        input_filepath = gr.Dropdown(
            choices=[
                ("train", "data/interim/train.parquet"),
                ("valid", "data/interim/val.parquet"),
                ("test", "data/interim/test.parquet"),
            ]
        )
        # reload_button = gr.Button("reload")
        input_filepath.select(load_input_data, inputs=[input_filepath])

    with gr.Tab("model check"):
        model_data_index = gr.Number(value=0, minimum=0, maximum=len(dataset))
        model_data_reload_button = gr.Button("reload")
        model_output_box = gr.Plot()
        model_output_dataframe = gr.DataFrame()

        model_data_index.change(
            fn=perform_inference,
            inputs=model_data_index,
            outputs=[model_output_box, model_output_dataframe],
        )


if __name__ == "__main__":
    demo.launch(share=False)
