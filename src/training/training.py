import torch
from tqdm import tqdm
import os

from .utils import collate_skip_stack_fn


def train_epoch_original(epoch_id, dataloader, model, optimizer, loss_fn, device='cpu', writer=None, verbose=True):
    model.train()

    if verbose:
        print(f'Running training for epoch {epoch_id}')

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    running_loss = 0.0
    train_instances = 0

    for i, batch in pbar:
        if batch is None:
            pbar.update(1)
            continue

        optimizer.zero_grad()

        real = batch['real_values'].to(device)
        user_id = batch['user_id'].to(device)
        predict_items = batch['predict_items'].to(device)
        timedeltas = batch['timedeltas'].to(device)
        weights = batch['weights'].to(device)

        output = model(user_id, predict_items, timedeltas, weights)

        loss = loss_fn(output, real)

        loss.backward()

        optimizer.step()

        loss_item = loss.item()
        pbar.update(1)

        if verbose:
            pbar.set_description(f'Batch loss: {loss_item}')
            train_instances += real.shape[0]
            running_loss += loss_item * real.shape[0]

        if writer is not None:
            global_step = epoch_id * len(dataloader) + i
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)

    total_loss = running_loss / train_instances
    if verbose:
        print(f'   epoch {epoch_id} loss: {total_loss}')


def eval_epoch_original(epoch_id, dataloader, model, loss_fn, metrics={}, device='cpu', writer=None, verbose=True):
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    running_loss = 0.0
    train_instances = 0

    running_metrics = {key: 0.0 for key, _ in metrics}

    with torch.no_grad():
        for i, batch in pbar:
            if batch is None:
                pbar.update(1)
                continue

            real = batch['real_values'].to(device)
            user_id = batch['user_id'].to(device)
            predict_items = batch['predict_items'].to(device)
            timedeltas = batch['timedeltas'].to(device)
            weights = batch['weights'].to(device)

            output = model(user_id, predict_items, timedeltas, weights)

            loss = loss_fn(output, real)

            loss_item = loss.item()

            metrics_dict = {}

            for key, value in metrics.items():
                metrics_dict[key] = value(output, real)

            if verbose:
                running_metrics = {key: value + metrics_dict[key] * real.shape[0] for key, value in
                                   running_metrics.items()}
                description = f'Batch loss: {loss_item:.04f}'.join(
                    [f';{key}: {value}' for key, value in metrics_dict.items()])
                pbar.set_description(description)
                train_instances += real.shape[0]
                running_loss += loss_item * real.shape[0]

            if writer is not None:
                global_step = epoch_id * len(dataloader) + i
                writer.add_scalar("Loss/train", loss.item(), global_step)
                for key, val in metrics.items():
                    writer.add_scalar(f'Metrics/{key}', val, global_step)

            pbar.update(1)

        if verbose:
            print(f'   epoch {epoch_id} loss: {running_loss / train_instances}'.join(
                [f';{key}: {value / train_instances}' for key, value in running_metrics.items()]))

        return running_loss / train_instances


def train_model(epochs_done, epoch_count, model, optimizer, dataloader_train, dataloader_val, loss_fn, metrics={},
                device='cpu', writer=None, verbose=False, save_best=False, save_last=1, save_dir='./checkpoints/'):
    curr_epoch_id = epochs_done

    best_loss = 100000.0

    while curr_epoch_id < epoch_count:
        train_epoch_original(curr_epoch_id, dataloader_train, model, optimizer, loss_fn, device, writer, verbose)

        if dataloader_val is not None:
            epoch_loss = eval_epoch_original(curr_epoch_id, dataloader_val, model, loss_fn, metrics, device, writer,
                                             verbose)

            if save_best and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save({
                    'epoch': curr_epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, f'{save_dir}checkpoint_best.pt')

            if save_last >= 1:
                torch.save({
                    'epoch': curr_epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                }, f'{save_dir}checkpoint_epoch{curr_epoch_id}.pt')

            if save_last >= 2 and os.path.exists(f'{save_dir}checkpoint_epoch{curr_epoch_id - save_last}'):
                os.remove(f'{save_dir}checkpoint_epoch{curr_epoch_id - save_last}')
        curr_epoch_id += 1


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
            loss_fn = torch.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"No such loss as {train_config['loss']} currently supported.")

    config = train_config['train']
    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=config['batch_size'],
                                                   num_workers=config['num_workers'], shuffle=config['shuffle'],
                                                   collate_fn=collate_skip_stack_fn)

    if val_data is not None:
        config = train_config['val']
        dataloader_val = torch.utils.data.DataLoader(val_data, batch_size=config['batch_size'],
                                                     num_workers=config['num_workers'], shuffle=config['shuffle'],
                                                     collate_fn=collate_skip_stack_fn)
    else:
        dataloader_val = None

    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) if log_dir is not None else None

    verbose = train_config['verbose'] if 'verbose' in train_config else False

    return {
        "epochs_done": epochs_done,
        "epoch_count": epoch_count,
        "model": model,
        "optimizer": optimizer,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "loss_fn": loss_fn,
        "device": train_config['device'],
        "writer": writer,
        "verbose": verbose,
    }
