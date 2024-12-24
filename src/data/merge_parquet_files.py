import pandas as pd
import argparse

def merge_parquet_files(parquet_path1, parquet_path2, output_path):
    # Load the Parquet files
    df1 = pd.read_parquet(parquet_path1)
    df2 = pd.read_parquet(parquet_path2)

    # Determine the higher resolution index
    high_res_index = df1.index.union(df2.index).sort_values()

    # Resample both DataFrames to the higher resolution index
    df1_resampled = df1.reindex(high_res_index).interpolate(method='linear')
    df2_resampled = df2.reindex(high_res_index).interpolate(method='linear')

    # Concatenate the DataFrames
    merged_df = pd.concat([df1_resampled, df2_resampled], axis=1)

    # Save the merged DataFrame to a new Parquet file
    merged_df.to_parquet(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two Parquet files.")
    parser.add_argument("input1", type=str, help="Path to the first input Parquet file.")
    parser.add_argument("input2", type=str, help="Path to the second input Parquet file.")
    parser.add_argument("output", type=str, help="Path to the output Parquet file.")
    args = parser.parse_args()

    merge_parquet_files(args.input1, args.input2, args.output)
