import mne
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.main import main_compute
from autoaugment.learning_curve import plot_result
mne.set_log_level("WARNING")


if __name__ == "__main__":
    train_sample, test_sample = get_epochs_data()
    dataset_args = {"transform_type": "raw (no transforms)"}

    # dl_args = {"model_type": "ShallowFBCSPNet",
    #            "batch_size": 64,
    #            "seed": None,
    #            "n_classes": 6,
    #            "lr": 0.00625,
    #            "weight_decay": 0,
    #            "n_epochs": 10,
    #            "n_cross_val": 3,
    #            "n_chans": int(train_sample[0][0].shape[0]),
    #            "input_window_samples": int(train_sample[0][0].shape[1])}

    dl_args = {"model_type": "ShallowFBCSPNet",
               "batch_size": 64,
               "seed": None,
               "n_classes": 6,
               "lr": 0.00625,
               "weight_decay": 0,
               "n_epochs": 100,
               "n_cross_val": 3,
               "n_chans": int(train_sample[0][0].shape[0]),
               "input_window_samples": int(train_sample[0][0].shape[1]),
               "train_split": True}

    # hf_args = {"model_type": "RandomForest",
    #            "n_cross_val": 3}
    sample_size_list = [0.01, 0.1, 0.25, 0.5, 1]
    saving_params = {"file_name": "model_v_2",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    main_compute([dl_args], [dataset_args],
                 train_sample, test_sample, sample_size_list,
                 saving_params)
    plot_result(saving_params, save_name="model_v1.3")
