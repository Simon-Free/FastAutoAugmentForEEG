import os
import getpass
import torch


transforms_args = {
    "n_transf": 1,
    "magnitude": 0.1,
    "masking_along_time":
        {"mask_value": 0.0,
         "axis": 2
         },
    "masking_along_frequency":
        {"mask_value": 0.0,
         "axis": 1
         },
    "em_decomposition":
        {"max_imfs": 12
         },
    "randaugment_transform_list":
        ["identity",
         "mask_along_time",
         "mask_along_frequency",
         "merge_two_signals",
         "delay_signal",
         "add_noise_to_signal"
         #  "merge_two_emd"
         ]
}

params_masking_random = {"mask_value": 0.0,
                         "mask_max_proportion": 0.1,
                         "axis": 2}


dl_dataset_args = {"transform_type": "raw (no transforms)",
                   "transform_list": [["identity"]]}


hf_dataset_args = {"transform_type": "raw (no transforms)",
                   "transform_list": [["identity_ml"]]}


dl_dataset_args_with_transforms = {
    "transform_type": "included masking transforms",
    "transform_list": [["identity"],
                       ["mask_along_time", "identity"],
                       ["mask_along_frequency", "identity"],
                       ["merge_two_signals"],
                       ["delay_signal"],
                       ["add_noise_to_signal"]
                       #    ["merge_two_emd"]
                       ],
}

hf_dataset_args_with_transforms = {
    "transform_type": "included masking transforms",
    "transform_list": [
        ["identity_ml"],
        ["mask_along_time", "identity_ml"],
        ["mask_along_frequency", "identity_ml"],
        ["merge_two_signals"],
        ["delay_signal"],
        ["add_noise_to_signal"]
        # ["merge_two_emd"]
    ]
}

shallow_args = {
    "model_type": "ShallowFBCSPNet",
    "batch_size": 64,
    "seed": None,
    "lr": 0.00625,
    "weight_decay": 0,
    "n_epochs": 100,
    "n_cross_val": 3,
    "criterion": torch.nn.CrossEntropyLoss,
    "device": "cuda:1",
    "patience": 5
}

sleepstager_args = {
    "model_type": "SleepStager",
    "batch_size": 256,
    "seed": None,
    "sfreq": 100,
    "lr": 0.002,
    "weight_decay": 0,
    "n_epochs": 50,
    "n_cross_val": 3,
    "criterion": torch.nn.CrossEntropyLoss,
    "device": "cuda:2",
    "patience": 5
}

hf_args = {"model_type": "RandomForest",
           "n_cross_val": 5,
           "n_estimators": 100,
           "random_state": 42}

sample_size_list = [0.01, 0.1, 0.25, 0.5, 1]

saving_params = {
    "result_dict_name": "main_result_dict",
    "main_save_folder": './results/'
}

if getpass.getuser() == "sfreybur":
    saving_params["main_save_folder"] = \
        "/storage/store/work/sfreybur/result_folder/"
else:
    default_dir = os.path.join(os.getcwd(), "result_folder")
