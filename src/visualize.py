import gradio as gr
import pandas as pd

def load_and_display_first_row(parquet_file):
    df = pd.read_parquet(parquet_file)
    first_row = df.iloc[0]
    return first_row.to_dict()

def main():
    interface = gr.Interface(
        fn=load_and_display_first_row,
        inputs=gr.Textbox(value="data/interim/infer_train.parquet", label="Parquet File Path"),
        outputs="json",
        title="Parquet Data Viewer",
        description="Displays the first row of the specified Parquet file."
    )
    return interface

if __name__ == "__main__":
    interface = main()
    interface.launch(share=False)
