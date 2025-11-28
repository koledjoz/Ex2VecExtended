import pytest
import torch

from data import init_dataset, get_dataset, get_dataset_type
from run.utils import collate_skip_stack_fn
from models import get_available_models
from ..utils import deep_update, assert_dicts_close



@pytest.mark.data
class TestData:
    @pytest.mark.parametrize("model_type", get_available_models())
    def test_dataset(self, model_type, train_data_config_factory, true_dataset_factory):
        train_data_config = train_data_config_factory(model_type)
        data_true = true_dataset_factory(model_type)

        init_dataset('train', train_data_config)
        train_data = get_dataset('train')

        # data_true = torch.load('./tests/files/test_data.pt')

        assert isinstance(train_data, get_dataset_type(model_type))
        assert len(train_data) == len(data_true)
        for i, (tested, true) in enumerate(zip(train_data, data_true)):
            print(f'Tested: {tested}, True: {true}')
            assert_dicts_close(tested, true, ignore={'predict_ts'})

    @pytest.mark.filterwarnings("ignore:This DataLoader will create.*:UserWarning")
    @pytest.mark.parametrize("model_type", get_available_models())
    @pytest.mark.parametrize("batch_size", [1, 16, 512, 1024])
    @pytest.mark.parametrize("num_workers", [0, 1, 2, 8, 32])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("max_padding", [4, 10, 50])
    def test_dataloader(self, model_type, train_data_config_factory, batch_size, num_workers, shuffle, max_padding, check_dataloader_shapes_factory):
        train_data_config = train_data_config_factory(model_type)
        check_dataloader_shapes = check_dataloader_shapes_factory(model_type)


        train_data_config['max_padding'] = max_padding
        init_dataset('train', train_data_config)
        train_data = get_dataset('train')

        dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                 num_workers=num_workers, shuffle=shuffle,
                                                 collate_fn=collate_skip_stack_fn)

        full_size = len(train_data)
        batch_counter = 0

        for batch in dataloader:
            if batch['real_values'].shape[0] != batch_size:
                batch_expected = full_size - batch_counter
            else:
                batch_expected = batch_size

            check_dataloader_shapes(batch, train_data, max_padding, batch_expected)

            batch_counter += batch_expected

        assert batch_counter == full_size
