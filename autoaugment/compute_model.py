import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import Subset
from braindecode.datasets.transform_classes import TransformSignal
from .models.deep_learning_models import get_deep_learning_model
from .models.handcrafted_features import get_randomforest
from .retrieve_data import get_sample
from .transforms.identity import identity
from .retrieve_data import create_label_index_dict
from .construct_transforms import construct_transforms


def initialize_model(model_args, train_sample, valid_dataset):
    if model_args["model_type"] in ["ShallowFBCSPNet", "SleepStager"]:
        model_args["n_classes"] = len(set(
            [train_sample[i][1] for i in range(len(train_sample))]))
        model_args["n_chans"] = int(train_sample[0][0].shape[0])
        model_args["input_window_samples"] = int(train_sample[0][0].shape[1])
        clf = get_deep_learning_model(model_args, valid_dataset)
    elif model_args["model_type"] == "RandomForest":
        clf = get_randomforest(model_args)
    else:
        raise ValueError('Boom!!!')
    return(clf)


def fit_model(model, model_args, train_dataset):

    if model_args["model_type"] == "RandomForest":
        x_train = np.concatenate([
            train_dataset[i][0].reshape(1, -1) for i
            in range(len(train_dataset))], axis=0)
        y_train = np.array([
            train_dataset[i][1] for i
            in range(len(train_dataset))])
        model.fit(x_train, y_train)
    else:
        y_train = np.array([data[1] for data in iter(train_dataset)])
        model.fit(train_dataset, y=y_train, epochs=model_args["n_epochs"])

    return(model)

# @memory.cache


def get_score(clf, model_args, test_dataset):

    if model_args["model_type"] == "RandomForest":
        x_test = np.concatenate([
            test_dataset[i][0].reshape(1, -1) for i
            in range(len(test_dataset))], axis=0)

        y_pred = clf.predict(x_test)
    else:
        y_pred = clf.predict(test_dataset)
    y_test = np.array([
        test_dataset[i][1] for i
        in range(len(test_dataset))])
    acc = accuracy_score(y_test, y_pred)
    # print(model_args["model_type"], " : ", str(acc))
    return(acc)


def create_transforms_and_subset(train_dataset, dataset_args, sample_size, transforms_args, i):
    train_subset = train_dataset
    train_dataset.change_transform_list([[TransformSignal(identity)]])
    # then, find what will be the labels of augmented data, without constructing transforms
    subset_aug_sample, subset_aug_labels = get_sample(train_dataset,
                                                      dataset_args["transform_list"],
                                                      sample_size,
                                                      random_state=i)
    # Define everything needed to construct transforms, even if "train_subset" will be replaced
    # afterwards.
    transforms_args["train_sample"] = train_subset
    transforms_args["label_index_dict"] = create_label_index_dict(
        subset_aug_sample, subset_aug_labels)
    # Constructs transforms
    dataset_args["constructed_transform_list"] = construct_transforms(
        dataset_args, transforms_args)
    # Update train dataset
    train_dataset.change_transform_list(
        dataset_args["constructed_transform_list"])
    train_subset = Subset(
        dataset=train_dataset,
        indices=subset_aug_sample)
    for transform in dataset_args["constructed_transform_list"]:
        for operation in transform:
            operation.params["train_sample"] = train_subset
    return dataset_args, train_subset
