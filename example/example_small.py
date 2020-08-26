import mne
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.main import main_compute
from autoaugment.learning_curve import plot_result
mne.set_log_level("WARNING")

def small_dl():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    dataset_args = {"transform_type": "raw (no transforms)"}
    model_args = {"model_type": "ShallowFBCSPNet",
                  "batch_size": 64,
                  "seed": None,
                  "n_classes": 6,
                  "lr": 0.00625,
                  "weight_decay": 0,
                  "n_epochs": 3,
                  "n_cross_val": 3,
                  "n_chans": int(train_sample[0][0].shape[0]),
                  "input_window_samples": int(train_sample[0][0].shape[1]),
                  "try": "foo"}


    sample_size_list = [0.01, 1]
    result_dict = main_compute([model_args], [dataset_args],
                               train_sample, test_sample, sample_size_list)
    plot_result(result_dict, save_name="model_v1")


def small_hf():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    dataset_args = {"transform_type": "raw (no transforms)"}
    model_args = {"model_type": "RandomForest",
                  "batch_size": 64,
                  "seed": None,
                  "n_classes": 6,
                  "lr": 0.00625,
                  "weight_decay": 0,
                  "n_epochs": 3,
                  "n_cross_val": 3,
                  "n_chans": int(train_sample[0][0].shape[0]),
                  "input_window_samples": int(train_sample[0][0].shape[1])}

    sample_size_list = [0.01, 0.05]

    saving_params = {"file_name": "small_dict",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    main_compute([model_args], [dataset_args],
                               train_sample, test_sample,
                               sample_size_list, saving_params)

    plot_result(saving_params, save_name="model_hf_small")


if __name__ == "__main__":
    small_hf()
