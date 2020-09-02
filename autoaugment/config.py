import mne
from braindecode.datasets.transform_classes import TransformSignal, TransformFFT
from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.transforms.masking import mask_along_axis_random


params_masking_random = {"mask_value": 0.0,
                         "mask_param": 10,
                         "axis": 2}



dataset_args = {"transform_type": "raw (no transforms)",
                "transform_list": [[TransformSignal(identity)]]}

shallow_args = {"model_type": "ShallowFBCSPNet",
                "batch_size": 64,
                "seed": None,
                "lr": 0.00625,
                "weight_decay": 0,
                "n_epochs": 100,
                "n_cross_val": 3,
                "try": "bar",
                "train_split": None,
                }

saving_params = {"result_dict_name": "main_result_dict",
                 "folder": {"sfreybur": "/storage/store/work/sfreybur/result_folder/"},
                 "user": None}

hf_args = {"model_type": "RandomForest",
           "n_cross_val": 5}

sample_size_list = [0.01, 0.1, 0.25, 0.5, 1]
