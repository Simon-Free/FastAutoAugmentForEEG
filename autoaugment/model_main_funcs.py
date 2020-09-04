from sklearn.metrics import accuracy_score
from .models.shallowfbcspnet import get_shallowfbcspnet
from .models.handcrafted_features import get_randomforest
from joblib import Memory
import numpy as np
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


def initialize_model(model_args, train_sample):
    if model_args["model_type"] == "ShallowFBCSPNet":
        model_args["n_classes"] = len(set([train_sample[i][1]
                                           for i in range(len(
                                               train_sample))]))
        model_args["n_chans"] = int(train_sample[0][0].shape[0])
        model_args["input_window_samples"] = int(train_sample[0][0].shape[1])
        clf = get_shallowfbcspnet(model_args)
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
