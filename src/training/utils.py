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


def save_training_state(save_dictionary, save_file):
    torch.save(save_dictionary, save_file)


def get_optimizer(optimizer_name):
    match optimizer_name:
        case "adam":
            return torch.optim.Adam
        case "sgd":
            return torch.optim.SGD
        case _:
            raise ValueError(f"No such optimizer as {optimizer_name} currently supported.")


def get_metric(metric_name):
    match metric_name:
        case 'cross_entropy':
            return torch.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"No such metric as {metric_name} currently supported.")
