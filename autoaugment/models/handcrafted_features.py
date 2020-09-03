from sklearn.ensemble import RandomForestClassifier


def get_randomforest(model_args):
	# XXX : model_args ???
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    return clf

