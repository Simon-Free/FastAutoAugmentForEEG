import mne

from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.config import hf_dataset_args, dl_dataset_args, \
    shallow_args, saving_params, hf_args, sample_size_list
mne.set_log_level("WARNING")


def test_small_shallownet():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    saving_params["result_dict_name"] = "small_result_dict"
    main_compute([shallow_args], [dl_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


def test_small_handcrafted_features():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    saving_params["result_dict_name"] = "small_result_dict"
    main_compute([hf_args], [hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


def test_small_full():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    saving_params["result_dict_name"] = "small_result_dict"
    main_compute([shallow_args, hf_args],
                 [dl_dataset_args, hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


if __name__ == "__main__":
    test_small_shallownet()
