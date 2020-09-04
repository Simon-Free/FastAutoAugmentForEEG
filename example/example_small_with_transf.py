import mne
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.retrieve_data import get_epochs_data
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
saving_params["file_name"] = "small_dict"
hf_args["n_cross_val"] = 3
shallow_args["n_epochs"] = 3


def small_test_shallownet():
                 "folder": "/storage/store/work/sfreybur/result_folder/"}

main_compute([model_args], [dataset_args],
             train_sample, test_sample,
             sample_size_list, saving_params)


if __name__ == "__main__":
    train_sample, test_sample = train_sample, test_sample = get_epochs_data(
        train_subjects=[1],
        test_subjects=[2],
        recording=[1])
    small_test_shallownet()
    plot_result(saving_params, save_name="model_bar")
