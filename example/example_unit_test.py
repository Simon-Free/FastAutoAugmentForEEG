from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.main import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from braindecode.datasets.transform_classes import TransformSignal, TransformFFT
import mne
mne.set_log_level("WARNING")


def dummy_test_shallownet():
    train_sample, test_sample = get_dummy_sample()
    dataset_args = {"transform_type": "raw (no transforms)",
                    "transform_list": [[TransformSignal(identity)]]}
    model_args = {"model_type": "ShallowFBCSPNet",
                  "batch_size": 64,
                  "seed": None,
                  "n_classes": len(set([train_sample[i][1]
                                        for i in range(len(train_sample))])),
                  "lr": 0.00625,
                  "weight_decay": 0,
                  "n_epochs": 3,
                  "n_cross_val": 3,
                  "n_chans": int(train_sample[0][0].shape[0]),
                  "input_window_samples": int(train_sample[0][0].shape[1]),
                  "try": "bar"}

    saving_params = {"file_name": "dummy_dict",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    sample_size_list = [1]
    main_compute([model_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)

    plot_result(saving_params, save_name="model_foo")


def dummy_test_handcrafted_features():
    train_sample, test_sample = get_dummy_sample()
    dataset_args = {"transform_type": "raw (no transforms)",
                    "transform_list": [[TransformSignal(identity_ml)]]}
    model_args = {"model_type": "RandomForest",
                  "n_cross_val": 3}
    saving_params = {"file_name": "dummy_dict_hf",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    sample_size_list = [1]
    main_compute([model_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


def full_test():

    train_sample, test_sample = get_dummy_sample()
    dataset_args = {"transform_type": "raw (no transforms)",
                    "transform_list": [[TransformSignal(identity)]]}
    dl_args = {"model_type": "ShallowFBCSPNet",
               "batch_size": 64,
               "seed": None,
               "n_classes": len(set([train_sample[i][1]
                                    for i in range(len(train_sample))])),
               "train_split": None,
               "lr": 0.00625,
               "weight_decay": 0,
               "n_epochs": 3,
               "n_cross_val": 3,
               "n_chans": int(train_sample[0][0].shape[0]),
               "input_window_samples": int(train_sample[0][0].shape[1]),
               "try": "bar"}
    hf_args = {"model_type": "RandomForest",
               "n_cross_val": 3}
    import ipdb; ipdb.set_trace()
    saving_params = {"file_name": "dummy_dict",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    sample_size_list = [1]
    main_compute([hf_args, dl_args], [dataset_args, dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)


if __name__ == "__main__":
    full_test()
