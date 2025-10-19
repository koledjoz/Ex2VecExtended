import pytest
import tempfile
import os
import torch

from ..utils import assert_dicts_close
from training.training import prepare_training, train_epoch, save_training_state
from utils import load_checkpoint
from models import get_available_models

@pytest.mark.loadingsaving
class TestLoadingSaving:
    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_model_save_load(self, model_type, model_factory, train_data_factory, training_config, device):
        model = model_factory(model_type)
        train_data = train_data_factory(model_type)

        epochs_to_try = 3
        training_config['device'] = device
        args = prepare_training(model, train_data, train_data, None, training_config, None)

        for _ in range(epochs_to_try):
            loss = train_epoch(args['epochs_done'], args['dataloader_train'], args['model'], args['optimizer'],
                                 args['loss_fn'], args['device'], args['writer'], args['verbose'])

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ckpt.pt")
            # save_checkpoint(ckpt_path, model, optimizer, epoch=5, loss=loss.item())
            save_training_state({
                'epoch': epochs_to_try,
                'model_state_dict': args['model'].state_dict(),
                'optimizer_state_dict': args['optimizer'].state_dict(),
                'loss': loss,
            }, ckpt_path)

            checkpoint = load_checkpoint(ckpt_path)


            new_args = prepare_training(model, train_data, train_data, checkpoint, training_config, None)

            # print(checkpoint, new_args)

            assert new_args['epochs_done'] == epochs_to_try
            assert pytest.approx(checkpoint['loss'], rel=1e-5) == loss

            # Model params should match exactly
            for p1, p2 in zip(args['model'].parameters(), new_args['model'].parameters()):
                assert torch.allclose(p1, p2, atol=1e-6)

            # --- Optimizer state ---
            orig_state = args['optimizer'].state_dict()
            new_state = new_args['optimizer'].state_dict()

            assert orig_state.keys() == new_state.keys()

            # Compare param_groups directly
            assert orig_state["param_groups"] == new_state["param_groups"]

            # Compare per-parameter state
            for pid, state in orig_state["state"].items():
                for key, value in state.items():
                    new_value = new_state["state"][pid][key]
                    if torch.is_tensor(value):
                        assert torch.allclose(value, new_value, atol=1e-6)
                    else:
                        assert value == new_value


