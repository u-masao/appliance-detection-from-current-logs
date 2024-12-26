import gradio as gr
import pandas as pd

def load_and_display_first_row(parquet_file):
    df = pd.read_parquet(parquet_file)
    first_row = df.iloc[0]
    return first_row.to_dict()

with gr.Blocks() as demo:
        gr.Markdown("# Parquet Data Viewer")
        gr.Markdown("Displays the first row of the specified Parquet file.")
        
        with gr.Row():
            input_box = gr.Textbox(value="data/interim/infer_train.parquet", label="Parquet File Path")
            output_box = gr.JSON(label="First Row Data")
        
        input_box.change(fn=load_and_display_first_row, inputs=input_box, outputs=output_box)
    
    def main():
        gr.Markdown("# Parquet Data Viewer")
        gr.Markdown("Displays the first row of the specified Parquet file.")
        
        with gr.Row():
            input_box = gr.Textbox(value="data/interim/infer_train.parquet", label="Parquet File Path")
            output_box = gr.JSON(label="First Row Data")
        
        input_box.change(fn=load_and_display_first_row, inputs=input_box, outputs=output_box)
    
        return demo

if __name__ == "__main__":
    interface = main()
    interface.launch(share=False)
