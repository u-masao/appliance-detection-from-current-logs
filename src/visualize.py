import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.models.dataset import TimeSeriesDataset, load_data
from src.models.model import load_model
from src.models.predict import predict

target_columns = ["watt_black", "watt_red", "watt_living", "watt_kitchen"]
model_filepath = "models/checkpoint/epoch_0009-trial_3.pth"
fraction = 1.00

input_df = pd.DataFrame()
infered_df = pd.DataFrame()
model = None
dataset = None


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


def create_plot(concat_df, append_df, title):

    # init figure
    plt.close()
    fig = plt.figure(figsize=(12, 5))
    fig.suptitle(f"actual vs predicted: {title}")

    # make axes
    gs = fig.add_gridspec(2, 4)
    full_ax = fig.add_subplot(gs[0, :])
    detail_axes = [
        fig.add_subplot(gs[1, x], sharey=full_ax)
        for x in range(len(target_columns))
    ]

    # 全体像
    full_ax.set_title("actual plot")
    for column in [target_columns + ["predict"]]:
        full_ax.plot(concat_df[column], label=column, alpha=0.8)
    full_ax.set_ylabel("[watt]")

    # 個別の予測値
    pred_columns = [f"pred_{x}" for x in target_columns]
    for index, (actual_column, pred_column) in enumerate(
        zip(target_columns, pred_columns)
    ):
        detail_axes[index].set_title(actual_column)
        detail_axes[index].plot(
            append_df[actual_column], label="actual", alpha=0.8
        )
        detail_axes[index].plot(
            append_df[pred_column], label="predict", alpha=0.8
        )
        detail_axes[index].set_ylabel("[watt]")

    # make legend and grid
    for ax in [full_ax] + detail_axes:
        ax.legend(loc="upper left")
        ax.set_xlabel("relative time [min]")
        ax.grid()

    # tune layout
    fig.tight_layout()
    return fig


def perform_inference(data_index: int, data_type: str):
    with torch.no_grad():
        x, y = dataset[data_index]
        zero_y = torch.zeros(y.size())
        tailx_y = x[-y.size()[0] :, 1:5]

        output, embed = model(x.unsqueeze(0), y.unsqueeze(0))
        output_zero, embed_zero = model(x.unsqueeze(0), zero_y.unsqueeze(0))
        output_tailx, embed_tailx = model(x.unsqueeze(0), tailx_y.unsqueeze(0))
        output_predict = predict(
            model, x.unsqueeze(0), model.config.output_sequence_length
        )

        concat_df, append_df = create_dataframes(x, y, output[0])
        concat_zero_df, append_zero_df = create_dataframes(
            x, y, output_zero[0]
        )
        concat_tailx_df, append_tailx_df = create_dataframes(
            x, y, output_tailx[0]
        )
        concat_predict_df, append_predict_df = create_dataframes(
            x, y, output_predict[0]
        )

        title_prefix = (
            f"{data_type} set, index {data_index}, {model_filepath}, "
        )

        fig = create_plot(
            concat_df,
            append_df,
            title_prefix + "teacher forcing (cheating)",
        )
        fig_zero = create_plot(
            concat_zero_df,
            append_zero_df,
            title_prefix + "tgt is zero vector",
        )
        fig_tailx = create_plot(
            concat_tailx_df,
            append_tailx_df,
            title_prefix + "tgt is repeated tail of X",
        )
        fig_predict = create_plot(
            concat_predict_df,
            append_predict_df,
            title_prefix + "tgt is predicted",
        )

    return (
        gr.Plot(value=fig),
        gr.Plot(value=fig_zero),
        gr.Plot(value=fig_tailx),
        gr.Plot(value=fig_predict),
    )


with gr.Blocks() as demo:
    gr.Markdown("# Parquet Data Viewer")

    with gr.Row():
        data_type = gr.Dropdown(
            choices=["train", "val", "test"], label="dataset type"
        )

    with gr.Tab("model check"):
        model_data_index = gr.Number(
            value=0, minimum=0, maximum=len(dataset), label="index of dataset"
        )
        # model_data_reload_button = gr.Button("reload")
        model_output_box = gr.Plot()
        model_output_box_zero = gr.Plot()
        model_output_box_tailx = gr.Plot()
        model_output_box_predict = gr.Plot()

    with gr.Tab("data check"):
        infered_data_index = gr.Number(
            value=0, minimum=0, maximum=len(infered_df)
        )
        infered_output_table = gr.DataFrame()

    data_type.select(
        load_input_data,
        inputs=[data_type],
        outputs=[infered_output_table],
    )
    model_data_index.change(
        fn=perform_inference,
        inputs=[model_data_index, data_type],
        outputs=[
            model_output_box,
            model_output_box_zero,
            model_output_box_tailx,
            model_output_box_predict,
        ],
    )
    infered_data_index.change(
        fn=take_infered_data,
        inputs=infered_data_index,
        outputs=[infered_output_table],
    )
if __name__ == "__main__":
    demo.launch(share=True)
