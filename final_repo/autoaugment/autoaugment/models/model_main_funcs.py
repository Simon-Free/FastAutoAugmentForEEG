import pandas as pd
from sklearn.metrics import accuracy_score
from .shallowfbcspnet import get_shallowfbcspnet
from .handcrafted_features import get_randomforest
from joblib import Memory
import numpy as np
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


def initialize_model(model_args, test_dataset):
    # check if GPU is available, if True chooses to use it

    if model_args["model_type"] == "ShallowFBCSPNet":
        clf = get_shallowfbcspnet(model_args, test_dataset)
        return(clf)
    if model_args["model_type"] == "RandomForest":
        clf = get_randomforest(model_args)
        return(clf)
    else:
        return(None)


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
        model.fit(train_dataset, y=None, epochs=model_args["n_epochs"])

    return(model)


# @memory.cache
def get_score(clf, model_args, test_dataset):
    if model_args["model_type"] == "RandomForest":
        x_test = np.concatenate([
            test_dataset[i][0].reshape(1, -1) for i
            in range(len(test_dataset))], axis=0)
        y_test = np.array([
            test_dataset[i][1] for i
            in range(len(test_dataset))])
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        print(acc)
        return(acc)
    else:
        results_columns = ['train_loss', 'valid_loss',
                           'train_accuracy', 'valid_accuracy']
        df = pd.DataFrame(clf.history[:, results_columns],
                          columns=results_columns,
                          index=clf.history[:, 'epoch'])
        return(df.tail(1)['valid_accuracy'].values[0])
