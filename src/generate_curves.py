import argparse

from run.predicting.predicting import prepare_predict, predict
from utils import load_config, load_checkpoint
from models import check_model, get_available_models, load_model
from data import check_dataset, get_available_datasets, get_dataset, init_dataset

# from training.training_original import prepare_training_original, train_model_original
# from training.training_extended import prepare_training_extended, train_model_extended

from run.training.training import prepare_training, train_model
from run.predicting.analysis import prepare_generate_curves, generate_curves

def main():
    parser = argparse.ArgumentParser(description='Generate predictions')

    parser.add_argument('--config_data', type=str, required=True, help='Path to the data config file.')
    parser.add_argument('--config_model', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--checkpoint_load', type=str, required=False,
                        help='The path to a checkpoint of the model to be used for predictions.')
    parser.add_argument('--save_path', type=str, required=True, help='The path to the file where the predictions should '
                                                                    'be saved')
    parser.add_argument('--top_k', type=int, default=50, help='The number of items that should be predicted.')
    parser.add_argument('--run_config', type=str, required=True, help='The path to the config file with the needed run config.')
    parser.add_argument('--start_time', type=int, default=1654041600)
    parser.add_argument('--end_time', type=int, default=1661990376)
    parser.add_argument('--n_steps', type=int, default=500)

    args = parser.parse_args()

    model_config = load_config(args.config_model)

    dataset_config = load_config(args.config_data)

    run_config = load_config(args.run_config)


    if not check_model(model_config):
        raise ValueError(f"No such model as {model_config['model_type']} available, please choose from the following "
                         f"list of available options: {get_available_models()}")

    if not check_dataset(dataset_config):
        raise ValueError(f"No such dataset as {dataset_config['dataset_type']} available, please choose from "
                         f"the following list of available options: {get_available_datasets()}")


    checkpoint = load_checkpoint(args.checkpoint_load)

    dataset_config['dataset_type'] = 'ExtendedAnalysis'
    dataset_config['sample_negative'] = 1

    init_dataset('test', dataset_config)

    # data =

    data = get_dataset('test')


    if 'n_users' not in model_config:
        raise ValueError('Model config needs to specify the number of users.')

    if 'n_items' not in model_config:
        raise ValueError('Model config needs to specify the number of items.')

    run_config['batch_size'] = 2
    run_config['num_workers'] = 8


    model = load_model(model_config, checkpoint=checkpoint)

    gen_curve_args = prepare_generate_curves(model, data, args.save_path, run_config, time_begin=args.start_time, time_end=args.end_time, n_steps=args.n_steps)

    generate_curves(**gen_curve_args)

# parser.add_argument('--start_time', type=int, default=1654041600)
#     parser.add_argument('--end_time', type=int, default=1661990376)
#     parser.add_argument('--n_steps', type=int, default=500)

if __name__ == "__main__":
    main()