import argparse

from utils import load_config, load_checkpoint
from models import check_model, get_available_models, load_model
from data import check_dataset, get_available_datasets, get_dataset, init_dataset

from training.training import prepare_training, train_model

def main():
    parser = argparse.ArgumentParser(description='Trains an Ex2Vec model.')

    parser.add_argument('--config_training', type=str, required=True, help='Path to the training config file.')
    parser.add_argument('--config_model', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--config_data_train', type=str, required=False,
                        help='Path to the training dataset config file.')
    parser.add_argument('--config_data_val', type=str, required=False,
                        help='Path to the validation dataset config file.')
    parser.add_argument('--checkpoint_load', type=str, required=False,
                        help='The path to a checkpoint if want to resume training.')
    parser.add_argument('--checkpoint_save_dir', type=str, required=False,
                        help='Directory into which checkpoints should be saved.')
    parser.add_argument('--save_best', action='store_true',
                        help='Enables saving the the best model based on the validation score. Needs for the '
                             'checkpoint save directory and validation dataset config parameters to be present.')
    parser.add_argument('--log_dir', type=str, required=False, help='The directory in which logs should be saved.')
    parser.add_argument('--save_last', type=int, required=False, help='The number of last checkpoints to be saved. '
                                                                      'This means the state of model and optimizer '
                                                                      'for last n epochs should be saved.')
    parser.add_argument('--save_dir', type=str, required=True, help='The location where the checkpoints of the best '
                                                                    'and last models should be saved. It needs to be '
                                                                    'a directory.')

    args = parser.parse_args()

    training_config = load_config(args.config_training)

    model_config = load_config(args.config_model)

    train_dataset_config = load_config(args.config_data_train)

    val_dataset_config = load_config(args.config_data_val) if args.config_data_val is not None else None

    if not check_model(model_config):
        raise ValueError(f"No such model as {model_config['model_type']} available, please choose from the following "
                         f"list of available options: {get_available_models()}")

    if not check_dataset(train_dataset_config):
        raise ValueError(f"No such dataset as {train_dataset_config['dataset_type']} available, please choose from "
                         f"the following list of available options: {get_available_datasets()}")

    if val_dataset_config is not None and not check_dataset(val_dataset_config):
        raise ValueError(
            f"No such dataset as {val_dataset_config['dataset_type']} available, please choose from the following "
            f"list of available options: {get_available_datasets()}")

    checkpoint = load_checkpoint(args.checkpoint_load)

    init_dataset('train', train_dataset_config)
    train_data = get_dataset('train')

    if val_dataset_config is not None:
        init_dataset('val', val_dataset_config)
        val_data = get_dataset('val')
    else:
        val_data = None

    if 'n_users' not in model_config:
        model_config['n_users'] = max(train_data.get_n_users(), val_data.get_n_users() if val_data is not None else 0)

    if 'n_items' not in model_config:
        model_config['n_items'] = max(train_data.get_n_items(), val_data.get_n_items() if val_data is not None else 0)

    model = load_model(model_config, checkpoint=checkpoint)

    train_args = prepare_training(model, train_data, val_data, checkpoint, training_config, args.log_dir)

    train_model(**train_args, save_best=args['save_best'], save_last=args['save_last'], save_dir=args['checkpoint_save_dir'])


if __name__ == "__main__":
    main()