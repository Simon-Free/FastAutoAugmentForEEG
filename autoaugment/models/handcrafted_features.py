from sklearn.ensemble import RandomForestClassifier


def get_randomforest(model_args):

    clf = RandomForestClassifier(
        model_args["n_estimators"], model_args["random_state"])
    return clf
