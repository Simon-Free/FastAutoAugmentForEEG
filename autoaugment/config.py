import os
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.transforms.masking import mask_along_axis_random
import getpass

params_masking_random = {"mask_value": 0.0,
                         "mask_max_proportion": 0.1,
                         "axis": 2}


dl_dataset_args = {"transform_type": "raw (no transforms)",
                   "transform_list": [[TransformSignal(identity)]]}

hf_dataset_args = {"transform_type": "raw (no transforms)",
                   "transform_list": [[TransformSignal(identity_ml)]]}

dl_dataset_args_with_transforms = {
    "transform_type": "included masking transforms",
    "transform_list": [[TransformSignal(identity)],
                       [TransformFFT(mask_along_axis_random,
                                     params_masking_random),
                        TransformSignal(identity)]]}

hf_dataset_args_with_transforms = {"transform_type": "included masking transforms",
                                   "transform_list": [
                                       [TransformSignal(identity_ml)],
                                       [TransformFFT(mask_along_axis_random,
                                                     params_masking_random),
                                        TransformSignal(identity_ml)]]}

shallow_args = {
    "model_type": "ShallowFBCSPNet",
    "batch_size": 64,
    "seed": None,
    "lr": 0.00625,
    "weight_decay": 0,
    "n_epochs": 100,
    "n_cross_val": 3,
    "try": "bar",
    "train_split": None,
}

hf_args = {"model_type": "RandomForest",
           "n_cross_val": 5,
           "n_estimators": 100,
           "random_state": 42}

sample_size_list = [0.01, 0.1, 0.25, 0.5, 1]

saving_params = {
    "result_dict_name": "main_result_dict"
}

if getpass.getuser() == "sfreybur":
    saving_params["main_save_folder"] = "/storage/store/work/sfreybur/result_folder/"
else:
    default_dir = os.path.join(os.getcwd(), "result_folder")
