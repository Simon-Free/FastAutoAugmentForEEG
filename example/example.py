import mne

from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.config import dl_dataset_args, hf_dataset_args, \
    shallow_args, saving_params, hf_args, sample_size_list
mne.set_log_level("WARNING")


def test_shallownet(train_sample, test_sample):
    main_compute([shallow_args], [dl_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


def test_handcrafted_features(train_sample, test_sample):
    main_compute([hf_args], [hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


def test_full(train_sample, test_sample):
    main_compute([shallow_args, hf_args],
                 [dl_dataset_args, hf_dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


if __name__ == "__main__":
    train_sample, test_sample = get_epochs_data()
    test_full()
    plot_result(saving_params)
