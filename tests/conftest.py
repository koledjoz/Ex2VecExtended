import pytest
import torch

from models.original.model import Ex2VecOriginal
from models.extended.model import Ex2VecExtended
from data import init_dataset, get_dataset



@pytest.fixture
def true_dataset_factory():
    def _make_true_dataset(model_type):
        if model_type == 'original':
            return torch.load('./tests/files/test_data_original.pt')
        elif model_type == 'extended':
            return torch.load('./tests/files/test_data_extended.pt')
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _make_true_dataset


@pytest.fixture
def train_data_config_factory():
    def _make_data_config(model_type):
        data_config = {"verbose": False, "data_path": "./tests/files/testing_data.parquet",
                       "usage_dict_path": "./tests/files/testing_data_dict.json",
                       "timedeltas_list_path": "./tests/files/testing_interactions.h5", "history_size": 4,
                       "sample_negative": -1, "max_padding": 4}
        if model_type == 'original':
            data_config['dataset_type'] = 'original'
        elif model_type == 'extended':
            data_config['dataset_type'] = 'extended'
            data_config['grouping_size'] = 1000
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")
        return data_config

    return _make_data_config


@pytest.fixture
def train_data_factory(train_data_config_factory):
    def _make_train_data(model_type):
        cfg = train_data_config_factory(model_type)
        init_dataset('train', cfg)
        return get_dataset('train')

    return _make_train_data


@pytest.fixture
def model_config():
    return {
        "n_users": 10,
        "n_items": 20,
        "latent_d": 8,
    }


# @pytest.fixture(params=get_available_models())
@pytest.fixture
def model_factory(model_config):
    def _make_model(model_type):
        if model_type == 'original':
            return Ex2VecOriginal(model_config)
        elif model_type == 'extended':
            return Ex2VecExtended(model_config)
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _make_model


def _original_check_shapes(batch, train_data, max_padding, batch_expected, *args, **kwargs):
    assert 'real_values' in batch.keys()
    assert 'user_id' in batch.keys()
    assert 'predict_items' in batch.keys()
    assert 'timedeltas' in batch.keys()
    assert 'weights' in batch.keys()
    assert batch['real_values'].shape == (batch_expected, train_data.get_n_items())
    assert batch['user_id'].shape == (batch_expected,)
    assert batch['predict_items'].shape == (batch_expected, train_data.get_n_items())
    assert batch['timedeltas'].shape == (batch_expected, train_data.get_n_items(), max_padding)
    assert batch['weights'].shape == (batch_expected, train_data.get_n_items(), max_padding)


def _extended_check_shapes(batch, train_data, max_padding, batch_expected, *args, **kwargs):
    assert 'real_values' in batch.keys()
    assert 'user_id' in batch.keys()
    assert 'predict_items' in batch.keys()
    assert 'timedeltas' in batch.keys()
    assert 'weights' in batch.keys()
    assert batch['real_values'].shape == (batch_expected, train_data.get_n_items())
    assert batch['user_id'].shape == (batch_expected,)
    assert batch['predict_items'].shape == (batch_expected, train_data.get_n_items())
    assert batch['timedeltas'].shape == (batch_expected, max_padding)
    assert batch['weights'].shape == (batch_expected, max_padding)


@pytest.fixture
def check_dataloader_shapes_factory():
    def _check_shapes(model_type):
        if model_type == 'original':
            return _original_check_shapes
        elif model_type == 'extended':
            return _extended_check_shapes
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _check_shapes


def _original_check_model(model):
    assert model.embedding_user.weight.shape == (model.n_users + 1, model.latent_d)
    assert model.embedding_item.weight.shape == (model.n_items + 1, model.latent_d)
    assert model.user_bias.weight.shape == (model.n_users + 1, 1)
    assert model.item_bias.weight.shape == (model.n_items + 1, 1)
    assert model.user_lamb.weight.shape == (model.n_users + 1, 1)

def _extended_check_model(model):
    assert model.embedding_user.weight.shape == (model.n_users + 1, model.latent_d)
    assert model.embedding_item.weight.shape == (model.n_items + 1, model.latent_d)
    assert model.user_bias.weight.shape == (model.n_users + 1, 1)
    assert model.item_bias.weight.shape == (model.n_items + 1, 1)
    assert model.user_lamb.weight.shape == (model.n_users + 1, 1)


@pytest.fixture
def check_model_loaded_factory():
    def _check_model(model_type):
        if model_type == 'original':
            return _original_check_model
        elif model_type == 'extended':
            return _extended_check_model
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _check_model


def _original_prepare_input(n_users, n_items, batch_size, max_pad, device):
    user_id = torch.randint(0, n_users, (batch_size,)).to(device)
    item_id = torch.randint(0, n_items, (batch_size, n_items)).to(device)
    timedeltas = torch.rand(batch_size, n_items, max_pad).to(device)
    weights = torch.rand(batch_size, n_items, max_pad).to(device)
    return {
        'user_id': user_id,
        'item_id': item_id,
        'timedeltas': timedeltas,
        'weights': weights
    }

 # def forward(self, user_index, pred_item_indices, history_item_indices, history_timedeltas, history_weights):

def _extended_prepare_input(n_users, n_items, batch_size, max_pad, device):
    user_id = torch.randint(0, n_users, (batch_size,)).to(device)
    item_id = torch.randint(0, n_items, (batch_size, n_items)).to(device)
    history_item_indices = torch.randint(0, n_items, (batch_size, max_pad)).to(device)
    timedeltas = torch.rand(batch_size, max_pad).to(device)
    weights = torch.rand(batch_size, max_pad).to(device)
    return {
        'user_index': user_id,
        'pred_item_indices': item_id,
        'history_item_indices': history_item_indices,
        'history_timedeltas': timedeltas,
        'history_weights': weights
    }


@pytest.fixture
def prepare_input_factory():
    def _prepare_input(model_type):
        if model_type == 'original':
            return _original_prepare_input
        elif model_type == 'extended':
            return _extended_prepare_input
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")
    return _prepare_input


def _original_check_output(output, batch_size, n_items):
    assert output.shape == (batch_size, n_items)

def _extended_check_output(output, batch_size, n_items):
    assert output.shape == (batch_size, n_items)


@pytest.fixture
def check_output_factory():
    def _check_output(model_type):
        if model_type == 'original':
            return _original_check_output
        elif model_type == 'extended':
            return _extended_check_output
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _check_output


def _original_check_scalars(model):
    for param in [model.global_lamb, model.alpha, model.beta, model.gamma, model.cutoff]:
        assert param.shape == torch.Size([]), f"{param} should be a scalar"
        assert param.requires_grad

def _extended_check_scalars(model):
    for param in [model.global_lamb, model.alpha, model.beta, model.gamma, model.cutoff, model.smooth, model.force]:
        assert param.shape == torch.Size([]), f"{param} should be a scalar"
        assert param.requires_grad


@pytest.fixture
def check_scalars_factory():
    def _check_scalars(model_type):
        if model_type == 'original':
            return _original_check_scalars
        elif model_type == 'extended':
            return _extended_check_scalars
        else:
            raise RuntimeError(f"Unknown parameter {model_type}")

    return _check_scalars


@pytest.fixture
def training_config():
    return {
        "epoch_count": 1,
        "learning_rate": 0.0001,
        "optimizer": "adam",
        "device": "cuda:0",
        "loss": "cross_entropy",
        "metrics": ["cross_entropy"],
        "train": {
            "batch_size": 32,
            "num_workers": 2,
            "shuffle": True
        },
        "val": {
            "batch_size": 32,
            "num_workers": 2,
            "shuffle": False
        }
    }
