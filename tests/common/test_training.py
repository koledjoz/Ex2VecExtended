import pytest
import torch
import copy
import math


from run.training.training import prepare_training, train_epoch, eval_epoch, train_model
from models import get_available_models
from ..utils import deep_update, assert_dicts_close



@pytest.mark.training
class TestTraining:
    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize("device", ['cpu', 'cuda'])
    @pytest.mark.parametrize('optimizer', ['adam', 'sgd'])
    def test_prepare_training(self, model_type, model_factory, train_data_factory, training_config, device, optimizer):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        overrides = {
            "device": device,
            "optimizer": optimizer,
        }
        cfg = deep_update(copy.deepcopy(training_config), overrides)

        args = prepare_training(model, train_data, train_data, None, cfg, None)

        assert args['epochs_done'] == 0
        assert args['epoch_count'] == 1
        if cfg['optimizer'] == 'adam':
            assert isinstance(args['optimizer'], torch.optim.Adam)
        elif cfg['optimizer'] == 'sgd':
            assert isinstance(args['optimizer'], torch.optim.SGD)

        assert isinstance(args['loss_fn'], torch.nn.CrossEntropyLoss)

    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_train_epoch_from_scratch(self, model_type, model_factory, train_data_factory, training_config, device):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        training_config['device'] = device
        args = prepare_training(model, train_data, train_data, None, training_config, None)

        loss = train_epoch(args['epochs_done'], args['dataloader_train'], args['model'], args['optimizer'],
                                    args['loss_fn'], args['device'], args['writer'], args['verbose'])
        assert isinstance(loss, float)
        assert math.isfinite(loss)
        assert loss > 0

    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    @pytest.mark.parametrize('metric', ['cross_entropy'])
    def test_evaluate_epoch(self, model_type, model_factory, train_data_factory, training_config, device, metric):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        training_config['device'] = device
        training_config['metrics'] = [metric]
        args = prepare_training(model, train_data, train_data, None, training_config, None)

        loss = eval_epoch(args['epochs_done'], args['dataloader_train'], args['model'],
                                   args['loss_fn'], args['metrics'], args['device'], args['writer'], args['verbose'])

        for metric, value in loss.items():
            assert isinstance(value, float)
            assert math.isfinite(value)
            assert value > 0

    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_model_weights_update(self, model_type, model_factory, train_data_factory, training_config, device):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        training_config['device'] = device
        args = prepare_training(model, train_data, train_data, None, training_config, None)

        initial_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        train_epoch(args['epochs_done'], args['dataloader_train'], args['model'], args['optimizer'],
                             args['loss_fn'], args['device'], args['writer'], args['verbose'])

        updated_params = [p.clone().detach() for p in model.parameters() if p.requires_grad]

        assert any(not torch.equal(p0, p1) for p0, p1 in zip(initial_params, updated_params))

    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_multi_epoch_learning(self, model_type, model_factory, train_data_factory, training_config, device):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        training_config['device'] = device
        args = prepare_training(model, train_data, train_data, None, training_config, None)

        losses = []
        losses.append(eval_epoch(args['epochs_done'], args['dataloader_train'], args['model'],
                                   args['loss_fn'], args['metrics'], args['device'], args['writer'], args['verbose'])['loss'])

        for _ in range(3):
            train_epoch(args['epochs_done'], args['dataloader_train'], args['model'], args['optimizer'],
                                 args['loss_fn'], args['device'], args['writer'], args['verbose'])
            losses.append(eval_epoch(args['epochs_done'], args['dataloader_train'], args['model'],
                                              args['loss_fn'], args['metrics'], args['device'], args['writer'],
                                              args['verbose'])['loss'])

        assert min(losses) < losses[0]

        for name, param in args['model'].named_parameters():
            assert torch.isfinite(param).all(), f"Parameter {name} contains NaN or Inf"

        for name, buf in args['model'].named_buffers():
            assert torch.isfinite(buf).all(), f"Buffer {name} contains NaN or Inf"
