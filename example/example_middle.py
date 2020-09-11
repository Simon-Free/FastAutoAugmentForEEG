import mne

from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.config import dl_dataset_args, \
    dl_dataset_args_with_transforms, \
    shallow_args, saving_params, sleepstager_args
mne.set_log_level("WARNING")

saving_params["result_dict_name"] = "middle_result_dict"
shallow_args["n_epochs"] = 50
shallow_args["n_cross_val"] = 1
sleepstager_args["n_epochs"] = 50
sleepstager_args["n_cross_val"] = 1
sample_size_list = [1]

if __name__ == "__main__":

    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=range(0, 10), valid_subjects=range(10, 15),
        test_subjects=range(15, 25),
        preprocessing=[])

    import ipdb
    ipdb.set_trace()

    dl_dataset_args["transform_type"] = "raw (no transforms)" \
        "+ no preprocessing"

    main_compute([shallow_args, sleepstager_args],
                 [dl_dataset_args, dl_dataset_args],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)

    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=range(0, 10), valid_subjects=range(10, 15),
        test_subjects=range(15, 25),
        preprocessing=["scaling"])

    dl_dataset_args["transform_type"] = "raw (no transforms)" \
        "+ scaling"

    main_compute([shallow_args, sleepstager_args],
                 [dl_dataset_args, dl_dataset_args],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)

    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=range(0, 10), valid_subjects=range(10, 15),
        test_subjects=range(15, 25),
        preprocessing=["scaling", "filtering"])

    dl_dataset_args["transform_type"] = "raw (no transforms)" \
        "+ scaling, filtering"
    dl_dataset_args_with_transforms["transform_type"] = \
        "masking + scaling, filtering"

    # run_handcrafted_features(train_sample, test_sample)
    main_compute([shallow_args, sleepstager_args,
                  shallow_args, sleepstager_args],
                 [dl_dataset_args, dl_dataset_args,
                  dl_dataset_args_with_transforms,
                  dl_dataset_args_with_transforms],
                 train_sample, valid_sample, test_sample,
                 sample_size_list, saving_params)
