import torch
import mne
from braindecode.datasets.transform_classes import TransformSignal
from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.config import params_masking, dataset_args,\
     shallow_args, saving_params, hf_args, sample_size_list


mne.set_log_level("WARNING")


def test_dummy_shallownet():
    train_sample, test_sample = get_dummy_sample()
    shallow_args["n_epochs"] = 3
    sample_size_list = [1]
    saving_params["result_dict_name"] = "dummy_dict"
    main_compute([shallow_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)
    

def test_dummy_handcrafted_features():
    train_sample, test_sample = get_dummy_sample()
    hf_args["n_cross_val"] = 3
    saving_params["result_dict_name"] = "dummy_dict"
    sample_size_list = [1]
    main_compute([hf_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


def test_full_dummy():

    train_sample, test_sample = get_dummy_sample()
    hf_args["n_cross_val"] = 3
    shallow_args["n_epochs"] = 3
    saving_params["result_dict_name"] = "dummy_dict"
    sample_size_list = [1]
    main_compute([shallow_args, hf_args], [dataset_args, dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


if __name__ == "__main__":
    test_full_dummy()
