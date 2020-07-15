import pandas as pd
from .shallowfbcspnet import get_shallowfbcspnet
from joblib import Memory
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


def initialize_model(model_args, test_dataset):
    # check if GPU is available, if True chooses to use it

    if model_args["model_type"] == "ShallowFBCSPNet":
        clf = get_shallowfbcspnet(model_args, test_dataset)
        return(clf)
    else:
        return(None)


# @memory.cache
def get_score(clf):
    results_columns = ['train_loss', 'valid_loss',
                       'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns],
                      columns=results_columns,
                      index=clf.history[:, 'epoch'])
    return(df.tail(1)['valid_accuracy'].values[0])
