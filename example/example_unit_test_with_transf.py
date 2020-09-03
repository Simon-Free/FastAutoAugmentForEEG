import torch
import mne
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.transforms.masking import mask_along_axis_random
from autoaugment.config import shallow_args, saving_params, \
    hf_args, params_masking_random
mne.set_log_level("WARNING")


dl_dataset_args = {"transform_type": "included masking transforms",
                   "transform_list": [
                       [TransformSignal(identity)],
                       [TransformFFT(mask_along_axis_random,
                                     params_masking_random),
                           TransformSignal(identity)]]}

hf_dataset_args = {"transform_type": "included masking transforms",
                   "transform_list": [
                       [TransformSignal(identity_ml)],
                       [TransformFFT(mask_along_axis_random,
                                     params_masking_random),
                        TransformSignal(identity_ml)]]}
sample_size_list = [1]
saving_params["file_name"] = "dummy_dict"
hf_args["n_cross_val"] = 3
shallow_args["n_epochs"] = 3


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
    dummy_test_shallownet()
    plot_result(saving_params)
