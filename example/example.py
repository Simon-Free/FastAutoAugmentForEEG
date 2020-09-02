import mne

from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.config import dataset_args,\
     shallow_args, saving_params, hf_args, sample_size_list
mne.set_log_level("WARNING")


def test_shallownet():
    train_sample, test_sample = get_epochs_data()
    main_compute([shallow_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


def test_handcrafted_features():
    train_sample, test_sample = get_epochs_data()
    main_compute([hf_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


def test_full():
    train_sample, test_sample = get_epochs_data()
    main_compute([shallow_args, hf_args],
                 [dataset_args, dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)
    plot_result(saving_params)


if __name__ == "__main__":
    test_full()