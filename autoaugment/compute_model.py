import numpy as np
from sklearn.metrics import accuracy_score
from .retrieve_data import get_sample
from .models.deep_learning_models import get_deep_learning_model
from .models.handcrafted_features import get_randomforest


def compute_experimental_result(model_args,
                                dataset_args,
                                train_dataset,
                                test_dataset,
                                sample_size):

    train_dataset.change_transform_list(dataset_args["transform_list"])
    score_list = []

    for i in range(model_args["n_cross_val"]):

        train_subset = get_sample(train_dataset,
                                  sample_size,
                                  random_state=i)
        model = initialize_model(model_args, train_subset)
        model = fit_model(model, model_args, train_subset)
        score_list.append(get_score(model, model_args, test_dataset))

    return score_list


def initialize_model(model_args, train_sample):
    if model_args["model_type"] in ["ShallowFBCSPNet", "ChambonSleepStager"]:
        model_args["n_classes"] = len(set([train_sample[i][1]
                                    for i in range(len(
                                        train_sample))]))
        model_args["n_chans"] = int(train_sample[0][0].shape[0])
        model_args["input_window_samples"] = int(train_sample[0][0].shape[1])
        clf = get_deep_learning_model(model_args)
        return clf
    if model_args["model_type"] == "RandomForest":
        clf = get_randomforest(model_args)
        return clf
    else:
        raise ValueError('Boom!!!')


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
