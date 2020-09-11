import mne
import torch
from autoaugment.tests.utils import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.config import dl_dataset_args, hf_dataset_args, \
    shallow_args, saving_params, hf_args, sleepstager_args

mne.set_log_level("WARNING")

shallow_args["n_epochs"] = 3
hf_args["n_cross_val"] = 3
sample_size_list = [1]
saving_params["result_dict_name"] = "dummy_dict"
shallow_args["criterion"] = torch.nn.CrossEntropyLoss


def dummy_shallownet(train_sample, valid_sample, test_sample):
    main_compute([shallow_args], [dl_dataset_args],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def dummy_handcrafted_features(train_sample, valid_sample, test_sample):
    main_compute([hf_args], [hf_dataset_args],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def dummy_sleepstagernet(train_sample, valid_sample, test_sample):
    main_compute([sleepstager_args], [dl_dataset_args],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def test_dummy_shallownet():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_shallownet(train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)


def test_dummy_handcrafted_features():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_handcrafted_features(train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)


def test_dummy_sleepstagernet():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_sleepstagernet(train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)
