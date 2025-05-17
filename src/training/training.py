import torch


def train_epoch_original(epoch_id, dataloader, model, optimizer, loss):
    return None


def eval_epoch_original(epoch_id, dataloader, model, metrics: dict):
    return None


def train_model(epochs_done, epoch_count, model, dataloader_train, dataloader_val, loss, device):
    return None


def prepare_training(model, train_data, val_data, checkpoint, train_config, log_dir=None):
    match train_config['optimizer']:
        case "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
        case "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=train_config['learning_rate'])
        case _:
            raise ValueError(f"No such optimizer as {train_config['optimizer']} currently supported.")

    epochs_done = 0
    epoch_count = train_config['epoch_count']

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']

    match train_config['loss']:
        case 'cross_entropy':
            loss = torch.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"No such loss as {train_config['loss']} currently supported.")

    with train_config['train'] as config:
        dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                                       num_workers=config['num_workers'], shuffle=config['shuffle'])

    if val_data is not None:
        with train_config['val'] as config:
            dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                                         num_workers=config['num_workers'], shuffle=config['shuffle'])
    else:
        dataloader_val = None

    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) if log_dir is not None else None

    return {
        "epochs_done": epochs_done,
        "epoch_count": epoch_count,
        "model": model,
        "datalaoder_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "loss": loss,
        "device": train_config['device'],
        "writer": writer
    }
