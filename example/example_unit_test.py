import torch
import mne
from braindecode.datasets.transform_classes import TransformSignal
from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.config import dl_dataset_args, hf_dataset_args, \
    shallow_args, saving_params, hf_args


mne.set_log_level("WARNING")

shallow_args["n_epochs"] = 3
hf_args["n_cross_val"] = 3
sample_size_list = [1]
saving_params["result_dict_name"] = "dummy_dict"


def test_dummy_shallownet(train_sample, test_sample):
    main_compute([shallow_args], [dl_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


def test_dummy_handcrafted_features(train_sample, test_sample):
    main_compute([hf_args], [hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


def test_full_dummy(train_sample, test_sample):
    main_compute([shallow_args, hf_args], [dl_dataset_args, hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


if __name__ == "__main__":
    train_sample, test_sample = get_dummy_sample()
    test_full_dummy()
    plot_result(saving_params)
