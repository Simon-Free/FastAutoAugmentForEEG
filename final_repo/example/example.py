import mne
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_model import compute_experimental_result
mne.set_log_level("WARNING")


if __name__ == "__main__":
    train_sample, test_sample = get_epochs_data(train_subjects=[0], test_subjects=[1])
    model_args = {"model_type": "ShallowFBCSPNet",
                  "batch_size": 64,
                  "seed": 20200220,
                  "n_classes": 6,
                  "lr": 0.00625,
                  "weight_decay": 0,
                  "n_epochs": 1,
                  "n_chans": train_sample[0][0].shape[0],
                  "input_window_samples": train_sample[0][0].shape[1]}
    sample_size = 0.1
    compute_experimental_result(model_args,
                                train_sample,
                                test_sample,
                                sample_size)
