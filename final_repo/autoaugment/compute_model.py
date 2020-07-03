from joblib import Memory
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)

@memory.cache
def compute_experimental_result(model_args, dataset_args):
    model = package.model_initialize(model_args)
    train_dataset, test_dataset = package.dataset_initialize(model_args)
    model.fit(train_dataset)
    score = package.get_score(model, test_dataset)
    return score