
import os
import pickle

from braindecode.datasets.transform_classes import TransformSignal

from .utils import update_saving_params
from .compute_model import initialize_model, get_score, fit_model, \
    create_transforms_and_subset
from .transforms.randaugment_transform import randaugment


def main_compute(model_args_list, dataset_args_list, transforms_args,
                 train_dataset, valid_dataset, test_dataset,
                 sample_size_list, saving_params):
    """
    Train every models given in entry, on their associated dataset.
    Store their validation accuracy on the test_dataset in a dict, and
    pickle it. Returns None.

    Parameters
    ----------
    model_args_list: dict
        contains all informations needed for model creation and training,
        keys needed depends of the model, see config.py.
    dataset_args_list: dict
        contains all informations needed for dataset creation and
        preprocessing.
    train_dataset: BaseConcatDataset
        dataset on which the model will be trained.
    valid_dataset: BaseConcatDataset
        dataset on which accuracy will be controlled at each epoch, for
        deep learning models
    test_dataset: BaseConcatDataset
        dataset on which the accuracy will be finally tested at the end
        of the training
    sample_size_list: list
        list of the proportions used to build the learning curve
    saving_params: dict
        informations useful for results saving

    Returns
    -------
    None
    """

    saving_params = update_saving_params(saving_params)
    result_dict_path = os.path.join(
        saving_params["result_dict_save_folder"],
        saving_params["result_dict_name"])

    try:
        with open(result_dict_path, 'rb') as handle:
            result_dict = pickle.load(handle)
    except (OSError, IOError):
        result_dict = {}

    for model_args, dataset_args in zip(model_args_list, dataset_args_list):
        key = (model_args["model_type"] + " + "
               + dataset_args["transform_type"])
        for sample_size in sample_size_list:
            print("computing model " + key +
                  " with sample size " + str(sample_size) + ".\n"
                  "transforms_list : " + str(dataset_args["transform_list"]))
            score = compute_experimental_result(model_args,
                                                dataset_args,
                                                transforms_args,
                                                train_dataset,
                                                valid_dataset,
                                                test_dataset,
                                                sample_size)
            if key not in result_dict.keys():
                result_dict[key] = {}
            result_dict[key][sample_size] = score

    with open(result_dict_path, 'wb') as handle:
        pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main_compute_with_randaugment(
        model_args_list, dataset_args_list, transforms_args,
        train_dataset,
        valid_dataset, test_dataset, sample_size_list, saving_params):
    """
    Train every models given in entry, on their associated dataset.
    Store their validation accuracy on the test_dataset in a dict, and
    pickle it. Returns None.

    Parameters
    ----------
    model_args_list: dict
        contains all informations needed for model creation and training,
        keys needed depends of the model, see config.py.
    dataset_args_list: dict
        contains all informations needed for dataset creation and
        preprocessing.
    train_dataset: BaseConcatDataset
        dataset on which the model will be trained.
    valid_dataset: BaseConcatDataset
        dataset on which accuracy will be controlled at each epoch, for
        deep learning models
    test_dataset: BaseConcatDataset
        dataset on which the accuracy will be finally tested at the end
        of the training
    sample_size_list: list
        list of the proportions used to build the learning curve
    saving_params: dict
        informations useful for results saving

    Returns
    -------
    None
    """

    saving_params = update_saving_params(saving_params)
    result_dict_path = os.path.join(
        saving_params["result_dict_save_folder"],
        saving_params["result_dict_name"])

    try:
        with open(result_dict_path, 'rb') as handle:
            result_dict = pickle.load(handle)
    except (OSError, IOError):
        result_dict = {}

    for model_args, dataset_args in zip(model_args_list, dataset_args_list):
        key = (model_args["model_type"] + " + "
               + dataset_args["transform_type"] + "with randaugment")
        for sample_size in sample_size_list:
            print("computing model " + key +
                  " with sample size " + str(sample_size))
            dict_score = {}
            for magnitude in transforms_args["magnitude_list"]:
                for n_transf in range(transforms_args["max_n_transf"]):
                    transforms_args["magnitude"] = magnitude
                    transforms_args["n_transf"] = n_transf
                    dataset_args["transform_list"] = [["randaugment"]]
                    score = compute_experimental_result(
                        model_args,
                        dataset_args,
                        transforms_args,
                        train_dataset,
                        valid_dataset,
                        test_dataset,
                        sample_size)
                    dict_score["magnitude : " + str(magnitude)
                               + ", n_transf : " + str(n_transf)] = score
            if key not in result_dict.keys():
                result_dict[key] = {}
            result_dict[key][sample_size] = dict_score

            with open(result_dict_path, 'wb') as handle:
                pickle.dump(result_dict, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)


def compute_experimental_result(model_args,
                                dataset_args,
                                transforms_args,
                                train_dataset,
                                valid_dataset,
                                test_dataset,
                                sample_size):

    score_list = []

    for i in range(model_args["n_cross_val"]):
        # First, initialize a raw dataset
        dataset_args, train_subset = create_transforms_and_subset(
            train_dataset, dataset_args, sample_size, transforms_args, i)
        # Construct train subset
        # Replace train subset as the reference dataframe for the transforms

        model = initialize_model(model_args, train_subset, valid_dataset)
        model = fit_model(model, model_args, train_subset)
        score_list.append(get_score(model, model_args, test_dataset))

    return score_list
