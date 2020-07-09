import mne
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.main import main_compute
from autoaugment.learning_curve import plot_result
mne.set_log_level("WARNING")


if __name__ == "__main__":
    train_sample, test_sample = get_epochs_data(train_subjects=[0],
                                                test_subjects=[1])
    dataset_args = {"transform_type": "raw (no transforms)"}
    sample_size = 0.1
    model_args = {"model_type": "ShallowFBCSPNet",
                  "batch_size": 64,
                  "seed": 20200220,
                  "n_classes": 6,
                  "lr": 0.00625,
                  "weight_decay": 0,
                  "n_epochs": 3,
                  "n_chans": int(train_sample[0][0].shape[0]),
                  "input_window_samples": int(train_sample[0][0].shape[1])}

    sample_size_list = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1]
    result_dict = main_compute([model_args], [dataset_args],
                               train_sample, test_sample, sample_size_list)
    plot_result(result_dict)
