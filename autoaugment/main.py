from .compute_model import compute_experimental_result
from pathlib import Path
import pickle
import os


def main_compute(model_args_list, dataset_args_list, train_dataset,
                 test_dataset, sample_size_list, saving_params):
    """Train every models given in entry, on their associated dataset.
    Return their validation accuracy on the test_dataset.

    Parameters
    ----------
    model_args_list: dict
        contains all informations needed for model creation and training,
        keys needed depends of the model.
    dataset_args_list: dict

    window_stride_samples: int
        stride between windows
    drop_last_window: bool
        whether or not have a last overlapping window, when
        windows do not equally divide the continuous signal

    Returns
    -------
    windows_datasets: BaseConcatDataset
        X and y transformed to a dataset format that is compatible with skorch
        and braindecode
    """

    Path(saving_params["folder"]).mkdir(parents=True, exist_ok=True)
    abs_result_dict_path = os.path.join(saving_params["folder"],
                                        saving_params["file_name"])
    try:
        with open(abs_result_dict_path, 'rb') as handle:
            result_dict = pickle.load(handle)
    except FileNotFoundError:
        result_dict = {}

    for model_args, dataset_args in zip(model_args_list, dataset_args_list):
        key = (model_args["model_type"] + " + "
               + dataset_args["transform_type"])
        for sample_size in sample_size_list:
            score = compute_experimental_result(model_args,
                                                train_dataset,
                                                test_dataset,
                                                sample_size)
            if key not in result_dict.keys():
                result_dict[key] = {}
            result_dict[key][sample_size] = score

    with open(abs_result_dict_path, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
