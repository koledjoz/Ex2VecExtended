import torch
import math
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

# from ..models.extended.model import Ex2VecExtended
# from ..models.original.model import Ex2VecOriginal

from .utils import collate_skip_stack_fn, save_training_state, get_optimizer, get_metric
# from utils import collate_skip_stack_fn



# # this method is not needed and will not be used as it is better to hide all of this in the model class that will be used
# def model_forward(model, batch, device):
#     if isinstance(model, Ex2VecOriginal):
#         user_id = batch['user_id'].to(device)
#         predict_items = batch['predict_items'].to(device)
#         timedeltas = batch['timedeltas'].to(device)
#         weights = batch['weights'].to(device)
#         return model(user_id, predict_items, timedeltas, weights)
#     elif isinstance(model, Ex2VecExtended):
#         user_id = batch['user_id'].to(device)
#         predict_items = batch['predict_items'].to(device)
#         history_items = batch['history_items'].to(device)
#         timedeltas = batch['timedeltas'].to(device)
#         weights = batch['weights'].to(device)
#         return model(user_id, predict_items, history_items, timedeltas, weights)
#     else:
#         raise TypeError(f'Unknown type passed. {type(model)} is not supported.')


def train_epoch(epoch_id, dataloader, model, optimizer, loss_fn, device='cpu', writer=None, verbose=True, log_step=-1,
                **kwargs):
    model.train()

    if verbose:
        print(f'Running training for epoch {epoch_id}')

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    running_loss = 0.0
    train_instances = 0
    losses = []

    for i, batch in pbar:
        if batch is None:
            pbar.update(1)
            continue

        optimizer.zero_grad()

        real = batch['real_values'].to(device)
        output = model.forward_batch(batch, device)

        loss = loss_fn(output, real)

        loss.backward()

        optimizer.step()
        if log_step > 0:
            losses.append(loss.detach())

        loss_item = loss.item()

        if log_step > 0 and verbose and i % log_step == 0:
            loss_item = torch.stack(losses).mean().item()
            pbar.set_description(f'Batch loss: {loss_item}')

        running_loss += loss_item * real.size(0)
        train_instances += real.size(0)

        if log_step > 0 and writer is not None and i % log_step == 0:
            global_step = epoch_id * len(dataloader) + i
            loss_item = torch.stack(losses).mean().item()
            writer.add_scalar("Loss/train", loss_item, global_step)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], global_step)

        if i % log_step == 0:
            losses = []

    total_loss = running_loss / train_instances
    if verbose:
        print(f'   epoch {epoch_id} loss: {total_loss}')

    return total_loss


def eval_epoch(epoch_id, dataloader, model, loss_fn, metrics={}, device='cpu', writer=None, verbose=True, **kwargs):
    model.eval()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=(not verbose))

    running_loss = 0.0
    train_instances = 0
    running_metrics = {key: 0.0 for key, _ in metrics.items()}

    with torch.no_grad():
        for i, batch in pbar:
            if batch is None:
                pbar.update(1)
                continue
            real = batch['real_values'].to(device)
            output = model.forward_batch(batch, device)
            loss = loss_fn(output, real).item()

            metrics_dict = {}
            for key, value in metrics.items():
                running_metrics[key] = running_metrics[key] + value(output, real).item() * real.shape[0]
                metrics_dict[key] = value(output, real).item()
            train_instances += real.shape[0]
            running_loss += loss * real.shape[0]

            if verbose:
                description = f'Batch loss: {loss:.04f}'.join(
                    [f';{key}: {value}' for key, value in metrics_dict.items()])
                pbar.set_description(description)

            pbar.update(1)

        if writer is not None:
            global_step = epoch_id * len(dataloader)
            writer.add_scalar("Loss/val", running_loss / train_instances, global_step)
            for key, val in running_metrics.items():
                writer.add_scalar(f'Metrics/{key}', val / train_instances, global_step)

        if verbose:
            print(f'   epoch {epoch_id} loss: {running_loss / train_instances}'.join(
                [f';{key}: {value / train_instances}' for key, value in running_metrics.items()]))

        running_metrics = {key: value/train_instances for key, value in running_metrics.items()}
        running_metrics['loss'] = running_loss / train_instances

        return running_metrics


def train_model(epochs_done, epoch_count, model, optimizer, dataloader_train, dataloader_val, loss_fn, metrics={},
                device='cpu', writer=None, verbose=False, save_best=False, save_last=1, save_dir='./checkpoints/',
                **kwargs):
    curr_epoch_id = epochs_done
    best_loss = math.inf

    while curr_epoch_id < epoch_count:
        train_epoch(curr_epoch_id, dataloader_train, model, optimizer, loss_fn, device, writer, verbose, **kwargs)

        if dataloader_val is not None:
            metrics_results = eval_epoch(curr_epoch_id, dataloader_val, model, loss_fn, metrics, device, writer, verbose, **kwargs)

            if save_best and metrics_results['loss'] < best_loss:
                best_loss = metrics_results['loss']
                save_training_state({
                    'epoch': curr_epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics_results['loss'],
                }, f'{save_dir}checkpoint_best.pt')

            if save_last >= 1:
                save_training_state({
                    'epoch': curr_epoch_id,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': metrics_results['loss'],
                }, f'{save_dir}checkpoint_epoch{curr_epoch_id}.pt')

            if save_last >= 2 and os.path.exists(f'{save_dir}checkpoint_epoch{curr_epoch_id - save_last}.pt'):
                os.remove(f'{save_dir}checkpoint_epoch{curr_epoch_id - save_last}.pt')

        curr_epoch_id += 1


def prepare_training(model, train_data, val_data, checkpoint, train_config, log_dir=None):
    print(f'Log dir can be found in {log_dir}')
    optimizer = get_optimizer(train_config['optimizer'])(model.parameters(), lr=train_config['learning_rate'])

    epochs_done = 0
    epoch_count = train_config['epoch_count']

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs_done = checkpoint['epoch']

    loss_fn = get_metric(train_config['loss'])

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

    writer = SummaryWriter(log_dir=log_dir) if log_dir is not None else None

    verbose = train_config['verbose'] if 'verbose' in train_config else False

    metrics = {}
    if "metrics" in train_config:
        for metric in train_config["metrics"]:
            metrics[metric] = get_metric(metric)

    return {
        "epochs_done": epochs_done,
        "epoch_count": epoch_count,
        "model": model.to(train_config['device']),
        "optimizer": optimizer,
        "dataloader_train": dataloader_train,
        "dataloader_val": dataloader_val,
        "loss_fn": loss_fn,
        "device": train_config['device'],
        "writer": writer,
        "verbose": verbose,
        "metrics": metrics
    }

