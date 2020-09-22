import numpy as np
import mne
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.config import dl_dataset_args, \
    dl_dataset_args_with_transforms, \
    transforms_args, \
    shallow_args, saving_params, sleepstager_args
mne.set_log_level("WARNING")

saving_params["result_dict_name"] = "middle_result_dict"
shallow_args["n_epochs"] = 50
shallow_args["n_cross_val"] = 5
sleepstager_args["n_epochs"] = 50
sleepstager_args["n_cross_val"] = 3
sample_size_list = [1]

if __name__ == "__main__":

    # train_sample, valid_sample, test_sample = get_epochs_data(
    #     train_subjects=range(0, 10), valid_subjects=range(10, 15),
    #     test_subjects=range(15, 25),
    #     preprocessing=[])

    # dl_dataset_args["transform_type"] = "raw (no transforms)" \
    #     "+ no preprocessing"

    # main_compute([shallow_args, sleepstager_args],
    #              [dl_dataset_args, dl_dataset_args],
    #              train_sample, valid_sample, test_sample,
    #              sample_size_list, saving_params)

    # train_sample, valid_sample, test_sample = get_epochs_data(
    #     train_subjects=range(0, 10), valid_subjects=range(10, 15),
    #     test_subjects=range(15, 25),
    #     preprocessing=["scaling"])

    # dl_dataset_args["transform_type"] = "raw (no transforms)" \
    #     "+ scaling"

    # main_compute([shallow_args, sleepstager_args],
    #              [dl_dataset_args, dl_dataset_args],
    #              train_sample, valid_sample, test_sample,
    #              sample_size_list, saving_params)

    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=range(0, 10), valid_subjects=range(10, 15),
        test_subjects=range(15, 25),
        preprocessing=["microvolt_scaling", "filtering"])

    dl_dataset_args_with_transforms["transform_list"] = [
        ["add_noise_to_signal"]]

    for magnitude in [0, 0.01, 0.1]:
        transforms_args["magnitude"] = magnitude
        dl_dataset_args_with_transforms["transform_type"] = "gaussian noise, "\
            "scaling, filtering" \
            "+ magnitude : " + str(magnitude)
        main_compute([sleepstager_args], [dl_dataset_args_with_transforms],
                     transforms_args, train_sample, valid_sample, test_sample,
                     sample_size_list, saving_params)

    # dl_dataset_args["transform_type"] = "raw (no transforms)" \
    #     "+ scaling, filtering"
    # dl_dataset_args_with_transforms["transform_type"] = \
    #     "masking + scaling, filtering"

    # run_handcrafted_features(train_sample, test_sample)
