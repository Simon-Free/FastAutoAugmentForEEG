def identity(datum, params={}):
    return datum


def identity_ml(datum, params={}):
    datum.X = datum.X.numpy()
    return(datum)
