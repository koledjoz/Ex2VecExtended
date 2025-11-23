import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# from ..models.extended.model import Ex2VecExtended
# from ..models.original.model import Ex2VecOriginal

from ..utils import collate_skip_stack_fn


def prepare_predict(model, checkpoint, data, output_path, run_config, top_k):
    model.load_state_dict(checkpoint['model_state_dict'])
    dataloader = torch.utils.data.DataLoader(data, batch_size=run_config['batch_size'],
                                             num_workers=run_config['num_workers'], shuffle=run_config['shuffle'],
                                             collate_fn=collate_skip_stack_fn)

    verbose = run_config['verbose'] if 'verbose' in run_config else False

    return {

        "model": model.to(run_config['device']),
        "dataloader": dataloader,
        "device": run_config['device'],
        "verbose": verbose,
        "top_k": top_k,
        "output_path": output_path
    }


def predict(model, dataloader, device, verbose, top_k, output_path):
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    prediction_cols = [f"pred_{i + 1}" for i in range(top_k)]

    all_batches = []

    with torch.no_grad():
        for i, batch in pbar:
            if batch is None:
                pbar.update(1)
                continue
            real = batch['real_values'].cpu().numpy
            model_result = model.forward_batch(batch, device).cpu().numpy()

            user_id = batch['user_id'].cpu().numpy()
            item_id = batch['predict_items'].cpu().numpy()
            ts = batch['predict_ts'].cpu().numpy()

            idx = real.argmax(axis=1)  # shape (B,)
            item_id = item_id[np.arange(len(item_id)), idx]

            predict_items = batch['predict_items'].cpu().numpy()

            # get the output of this
            idx = np.argsort(-model_result, axis=1)
            predict_items = np.take_along_axis(predict_items, idx, axis=1)[:, :top_k]

            # lets get the output

            u = np.squeeze(user_id)[:, None]  # -> (B, 1)
            i = np.squeeze(item_id)[:, None]  # -> (B, 1)
            t = np.squeeze(ts)[:, None]  # -> (B, 1)
            c = np.squeeze(predict_items)  # -> (B, top_k)

            data = np.concatenate([u, i, t, c], axis=1)

            batch_df = pd.DataFrame(data, columns=["userId", "trackId", "ts"] + prediction_cols)
            all_batches.append(batch_df)

            pbar.update(1)

    df_preds = pd.concat(all_batches, ignore_index=True)
    df_preds.to_csv(output_path)
