import torch


def collate_skip_stack_fn(batch):
    # Remove None entries
    batch = [x for x in batch if x is not None]

    if not batch:
        return None  # Signal to skip this batch

    # Stack each field in the batch
    collated_batch = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = torch.stack([sample[key] for sample in batch])

    return collated_batch
