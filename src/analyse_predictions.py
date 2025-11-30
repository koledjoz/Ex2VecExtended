import argparse
import pandas as pd
import os


def get_acc_at_inter(dataframe):
    df_new = dataframe.copy(deep=True)

    for col in [x for x in df_new.columns if 'pred_' in x]:
        df_new[col] = (df_new[col] == df_new['trackId']).astype(int)

    wanted_cols = [x for x in df_new.columns if 'pred_' in x]

    df_new[wanted_cols] = df_new[wanted_cols].cumsum(axis=1)

    df_new["interaction_num"] = df_new.groupby(["userId", "trackId"]).cumcount() + 1

    result_list = []

    for col in wanted_cols:
        temp = (
            df_new.groupby("interaction_num")[col]
            .agg(accuracy="mean", count="size")
            .reset_index()
        )
        temp["col_name"] = col
        result_list.append(temp)

    accuracy = pd.concat(result_list, ignore_index=True)
    return accuracy





def main():
    parser = argparse.ArgumentParser(description='Analyse predictions')
    parser.add_argument('--predictions_path', type=str, required=True, help='Path to the results directory.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the analysis directory.')
    parser.add_argument('--full_data', type=str, required=True, help='Path to the full data parquet file.')


    args = parser.parse_args()

    input_dir = args.predictions_path
    output_dir = args.output_path

    df_full = pd.read_csv(args.full_data)
    item_counts = df_full['track_id'].value_counts().rename_axis('track_id').reset_index(name='total_count')

    model_dirs = os.listdir(input_dir)
    for directory in model_dirs:
        path = os.path.join(input_dir, directory)
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)

            output_file_acc_at_k = os.path.join(output_dir, 'acc_at_k', directory, file)

            df = pd.read_csv(file_path)
            out = get_acc_at_inter(df)
            out.to_csv(output_file_acc_at_k, index=False)
            print(f'Successfully processed {file_path}, calculating acc_at_k')

            pred_cols = [col for col in df.columns if col.startswith("pred_")]

            accuracy_df = (
                df
                .assign(**{f"acc_{col}": (df[col] == df["trackId"]).astype(int) for col in pred_cols})
                .groupby("trackId")[[f"acc_{c}" for c in pred_cols]]
                .mean()
                .reset_index()
            )

            final_df = item_counts.merge(accuracy_df.rename(columns={'trackId': 'track_id'}), on="track_id", how="left")

            output_file_acc_at_pop = os.path.join(output_dir, 'acc_at_pop', directory, file)
            final_df.to_csv(output_file_acc_at_pop, index=False)

            print(f'Successfully processed {file_path}, calcualting acc_at_pop')




if __name__ == "__main__":
    main()