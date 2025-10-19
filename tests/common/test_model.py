import pytest
import torch
from flaky import flaky

from models import get_available_models
from models import get_model_type



@pytest.mark.model
class TestModel:
    @pytest.mark.parametrize("model_type", get_available_models())
    def test_load_model(self, model_type, model_factory, check_model_loaded_factory):
        model = model_factory(model_type)
        check_model_loaded = check_model_loaded_factory(model_type)

        assert isinstance(model, get_model_type(model_type))
        check_model_loaded(model)


    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128, 512, 1024])
    @pytest.mark.parametrize("max_pad", [25, 50, 100, 500])
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_forward_output_shapes(self, model_type, model_factory, batch_size, max_pad, prepare_input_factory, check_output_factory, device):
        model = model_factory(model_type).to(device)
        get_input = prepare_input_factory(model_type)
        check_output = check_output_factory(model_type)

        n_users, n_items = model.n_users, model.n_items

        inputs = get_input(n_users, n_items, batch_size, max_pad, device)
        out = model(**inputs)

        check_output(out, batch_size, n_items)


    @pytest.mark.parametrize("model_type", get_available_models())
    def test_parameters_have_expected_shapes(self, model_type, model_factory, check_scalars_factory):
        model = model_factory(model_type)
        check_scalars = check_scalars_factory(model_type)

        check_scalars(model)

    @flaky(max_runs=10, min_passes=10)
    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize("batch_size", [1, 4, 16, 128, 512, 1024])
    @pytest.mark.parametrize("max_pad", [25, 50, 100, 500])
    @pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
    def test_forward_runs_and_gradients(self, model_type, model_factory, batch_size, max_pad, prepare_input_factory, device):
        model = model_factory(model_type).to(device)
        get_input = prepare_input_factory(model_type)

        n_users, n_items, latent_d = model.n_users, model.n_items, model.latent_d

        inputs = get_input(n_users, n_items, batch_size, max_pad, device)
        out = model(**inputs)

        loss = out.mean()
        loss.backward()

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = p.grad
            assert g is not None, f"No grad for {name}"
            assert torch.isfinite(g).all(), f"NaN/Inf in gradient of {name}"

            # max_abs = g.abs().max().item()
            # assert max_abs <= 1e3, f"Exploding gradient {max_abs} in {name}"

        # Check that all parameters have gradient
        # grads = [p.grad for p in model.parameters() if p.requires_grad]
        # assert all(g is not None for g in grads), "Some gradients were not computed"
        #
        # for g in grads:
        #     assert torch.isfinite(g).all(), "Gradient contains NaN or Inf"
        #
        #     # Check values are in a reasonable range (tune threshold as needed)
        #     max_abs = g.abs().max().item()
        #     assert max_abs <= 1e4, f"Exploding gradient detected: {max_abs} at parameter"


