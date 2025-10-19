import pandas as pd
import numpy as np
import argparse
import h5py
import os
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)


def save_user_track_interactions_to_hdf5(df: pd.DataFrame, h5_path: str):
    """
    Given a dataframe with columns: user_id, track_id, timestamp,
    store grouped interactions into HDF5 with flat + offset layout.
    """
    assert set(df.columns) >= {"user_id", "track_id", "ts"}, "Missing required columns"

    # Group by (user_id, track_id)
    grouped = df.sort_values(["user_id", "track_id", "ts"]).groupby(["user_id", "track_id"])

    # Prepare storage arrays
    timestamps_flat = []
    offsets = []
    user_item = []

    logging.info('Preparing the flat arrays')

    current_offset = 0
    for (user, item), group in tqdm(grouped):
        ts = group["ts"].to_numpy(dtype=np.int64)
        timestamps_flat.append(ts)
        length = len(ts)
        offsets.append((current_offset, length))
        user_item.append((user, item))
        current_offset += length

    # Convert to flat arrays
    timestamps_flat = np.concatenate(timestamps_flat)
    offsets = np.array(offsets, dtype=np.int64)
    user_item = np.array(user_item, dtype=np.int64)

    logging.info('Saving to h5 file')

    # Save to HDF5
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("timestamps_flat", data=timestamps_flat, compression="gzip", chunks=True)
        f.create_dataset("offsets", data=offsets, compression="gzip", chunks=True)
        f.create_dataset("user_item", data=user_item, compression="gzip", chunks=True)

    logging.info(f"Saved {len(user_item)} user-item pairs, {len(timestamps_flat)} timestamps to {h5_path}")



def main():
    parser = argparse.ArgumentParser(description='Prepares dataset splits and configs for training of all Ex2Vec based models.')

    parser.add_argument('--data', type=str, required=True, help='The path to interactions parquet file containing the interactions.')
    parser.add_argument('--split_count', type=int, required=True, help='The number of splits to use for validation.')
    parser.add_argument('--data_directory', type=str, required=True, help='The directory to save the files.')
    parser.add_argument('--test_items', type=int, default=2, help='The number of items per user to be used in the test dataset.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='The ratio of items per user to be used in the validation set.')

    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    # save test dataset

    logging.info('Creating paths for test dataset')

    # check existence of data_directory
    if not os.path.exists(args.data_directory):
        os.makedirs(args.data_directory)

    test_dir_path = os.path.join(args.data_directory, 'test')
    if not os.path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    logging.info('Saving interactions for test dataset')

    save_user_track_interactions_to_hdf5(df, os.path.join(test_dir_path, 'interactions.h5'))

    logging.info('Sampling the test dataset json')

    sampled = (
        df[['user_id', 'track_id']].drop_duplicates()
        .groupby('user_id', group_keys=False)
        .apply(lambda x: x.sample(n=min(args.test_items, len(x))))
        .reset_index(drop=True)
    )

    sampled_dict = sampled.groupby('user_id')['track_id'].apply(list).to_dict()

    logging.info('Saving the sampled test dataset json')

    with open(os.path.join(test_dir_path, 'test_dict.json'), "w") as f:
        json.dump(sampled_dict, f, indent=4)

    merged = df.merge(sampled, on=['user_id', 'track_id'], how='left', indicator=True)

    df = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

    # this will be used for the next dataset including the rest of the data, so lets goooooooooooooo

    logging.info('Creating paths for validation and training datasets')

    val_dir_path = os.path.join(args.data_directory, 'val')
    if not os.path.exists(val_dir_path):
        os.makedirs(val_dir_path)

    save_user_track_interactions_to_hdf5(df, os.path.join(val_dir_path, 'interactions.h5'))

    train_dir_path = os.path.join(args.data_directory, 'train')
    if not os.path.exists(train_dir_path):
        os.makedirs(train_dir_path)



    for split_num in range(args.split_count):

        logging.info(f'[Split {split_num}]: creating paths')


        split_val_path = os.path.join(val_dir_path, f'split_{split_num}')
        split_train_path = os.path.join(train_dir_path, f'split_{split_num}')

        if not os.path.exists(split_val_path):
            os.makedirs(split_val_path)

        if not os.path.exists(split_train_path):
            os.makedirs(split_train_path)

        logging.info(f'[Split {split_num}]: Sampling the validation dataset json')

        sampled = (
            df[['user_id', 'track_id']].drop_duplicates()
            .groupby('user_id', group_keys=False)
            .apply(lambda x: x.sample(n=max(int(args.val_ratio * len(x)), 1)))
            .reset_index(drop=True)
        )


        sampled_dict = sampled.groupby('user_id')['track_id'].apply(list).to_dict()

        logging.info(f'[Split {split_num}]: Saving the validation dataset json')

        with open(os.path.join(split_val_path, 'val_dict.json'), "w") as f:
            json.dump(sampled_dict, f, indent=4)

        merged = df.merge(sampled, on=['user_id', 'track_id'], how='left', indicator=True)

        df_new = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

        logging.info(f'[Split {split_num}]: Saving the training interactions')

        save_user_track_interactions_to_hdf5(df_new, os.path.join(split_train_path, 'interactions.h5'))

        sampled_dict = df_new.groupby('user_id')['track_id'].apply(list).to_dict()

        logging.info(f'[Split {split_num}]: Saving the training dataset json')

        with open(os.path.join(split_train_path, 'train_dict.json'), "w") as f:
            json.dump(sampled_dict, f, indent=4)


if __name__ == "__main__":
    main()
