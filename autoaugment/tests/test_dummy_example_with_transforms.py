import mne
from autoaugment.tests.utils import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.config import shallow_args, saving_params, \
    dl_dataset_args_with_transforms, hf_dataset_args_with_transforms, \
    hf_args, sleepstager_args
mne.set_log_level("WARNING")


sample_size_list = [1]
saving_params["file_name"] = "dummy_dict"
hf_args["n_cross_val"] = 3
shallow_args["n_epochs"] = 3


def dummy_shallownet_with_transf(train_sample, valid_sample, test_sample):

    main_compute([shallow_args], [dl_dataset_args_with_transforms],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def dummy_handcrafted_features_with_transf(train_sample, valid_sample,
                                           test_sample):
    main_compute([hf_args], [hf_dataset_args_with_transforms],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def dummy_sleepstagernet_with_transf(train_sample, valid_sample, test_sample):
    main_compute([sleepstager_args], [dl_dataset_args_with_transforms],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)


def test_dummy_shallownet_with_transf():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_shallownet_with_transf(train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)


def test_dummy_handcrafted_features_with_transf():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_handcrafted_features_with_transf(
        train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)


def test_dummy_sleepstagernet():
    train_sample, valid_sample, test_sample = get_dummy_sample()
    dummy_sleepstagernet_with_transf(train_sample, valid_sample, test_sample)
    plot_result(saving_params)
    assert(True)


if __name__ == "__main__":
    test_dummy_shallownet_with_transf()
