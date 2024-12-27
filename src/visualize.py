import gradio as gr
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

input_filepath = "data/interim/infer_train.parquet"
infer_df = pd.read_parquet(input_filepath)
feature_df = pd.read_parquet("data/interim/train.parquet").iloc[[0]]
target_columns = ["watt_black", "watt_red", "watt_kitchen", "watt_living"]


def create_dataframes(row_number):
    sr = infer_df.iloc[row_number]
    train_df = pd.DataFrame(
        sr[sr.index.str.startswith("train_")].values.reshape(-1, 12),
        columns=feature_df.drop("gap", axis=1).columns,
    )
    actual_df = pd.DataFrame(
        sr[sr.index.str.startswith("actual_")].values.reshape(
            -1, len(target_columns)
        ),
        columns=target_columns,
    )
    pred_df = pd.DataFrame(
        sr[sr.index.str.startswith("pred_")].values.reshape(
            -1, len(target_columns)
        ),
        columns=target_columns,
    )
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


def load_and_display_row(row_number):
    concat_df, append_df = create_dataframes(row_number)
    fig = create_plot(concat_df, append_df)
    return gr.Plot(value=fig), gr.DataFrame(concat_df)


with gr.Blocks() as demo:
    gr.Markdown("# Parquet Data Viewer")

    row_number = gr.Number(value=0, minimum=0, maximum=len(infer_df))
    output_box = gr.Plot()
    output_dataframe = gr.DataFrame()

    row_number.change(
        fn=load_and_display_row,
        inputs=row_number,
        outputs=[output_box, output_dataframe],
    )

if __name__ == "__main__":
    demo.launch(share=False)
