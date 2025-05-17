import torch


def train_epoch_original(epoch_id, dataloader, model, optimizer, loss):
    return None


def eval_epoch_original(epoch_id, dataloader, model, metrics: dict):
    return None


def train_model(epochs_done, epoch_count, model, dataloader_train, dataloader_val, loss, device):
    return None


def prepare_training(model, train_data, val_data, checkpoint, train_config):
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

    losses = []

    for i, loss in enumerate(train_config['loss']):
        match loss:
            case 'cross_entropy':
                losses.append(torch.nn.CrossEntropyLoss())
            case _:
                raise ValueError(f"No such loss as {loss} currently supported.")

    with train_config['train'] as config:
        dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                                       num_workers=config['num_workers'], shuffle=config['shuffle'])

    if val_data is not None:
        with train_config['val'] as config:
            dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                                         num_workers=config['num_workers'], shuffle=config['shuffle'])
    else:
        dataloader_val = None

    return epochs_done, epoch_count, model, dataloader_train, dataloader_val, losses, train_config['device']
