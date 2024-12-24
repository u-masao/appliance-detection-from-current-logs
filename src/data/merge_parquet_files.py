import pandas as pd

def merge_parquet_files(parquet_path1, parquet_path2, output_path):
    # Load the Parquet files
    df1 = pd.read_parquet(parquet_path1)
    df2 = pd.read_parquet(parquet_path2)

    # Determine the higher resolution index
    if df1.index.freq < df2.index.freq:
        high_res_index = df1.index
    else:
        high_res_index = df2.index

    # Resample both DataFrames to the higher resolution index
    df1_resampled = df1.reindex(high_res_index).interpolate(method='linear')
    df2_resampled = df2.reindex(high_res_index).interpolate(method='linear')

    # Concatenate the DataFrames
    merged_df = pd.concat([df1_resampled, df2_resampled], axis=1)

    # Save the merged DataFrame to a new Parquet file
    merged_df.to_parquet(output_path)

if __name__ == "__main__":
    merge_parquet_files(
        "data/interim/env_temp.parquet",
        "data/interim/star_watt.parquet",
        "data/processed/merged_data.parquet"
    )
