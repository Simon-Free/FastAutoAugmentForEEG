from sklearn.ensemble import RandomForestClassifier


def get_randomforest(model_args):

    clf = RandomForestClassifier(
        n_estimators=model_args["n_estimators"],
        random_state=model_args["random_state"])
    return clf
