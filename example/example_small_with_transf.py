from autoaugment.retrieve_data import get_epochs_data
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.transforms.masking import mask_along_axis_random
from braindecode.datasets.transform_classes import TransformSignal, TransformFFT
import mne
mne.set_log_level("WARNING")


def small_test_shallownet():
    train_sample, test_sample = train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])

    params_masking = {"mask_value": 0.0,
                      "mask_param": 10,
                      "axis": 2}
    dataset_args = {"transform_type": "mask along axis",
                    "transform_list": [
                        [TransformSignal(identity)],
                        [TransformFFT(mask_along_axis_random, params_masking),
                         TransformSignal(identity)]]}

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
                  "try": "bar",
                  "train_split": None,
                  }
    sample_size_list = [0.01, 1]

    saving_params = {"file_name": "small_dict",
                     "folder": "/storage/store/work/sfreybur/result_folder/"}

    main_compute([model_args], [dataset_args],
                 train_sample, test_sample,
                 sample_size_list, saving_params)

    plot_result(saving_params, save_name="model_bar")


if __name__ == "__main__":
    small_test_shallownet()
