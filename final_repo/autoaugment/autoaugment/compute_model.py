from joblib import Memory
from .models.model_main_funcs import initialize_model, get_score
from .retrieve_data import get_sample
cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


@memory.cache
def compute_experimental_result(model_args,
                                train_dataset,
                                test_dataset,
                                sample_size):

    score_list = []
    for i in range(10):
        model = initialize_model(model_args, test_dataset)
        train_subset = get_sample(train_dataset, sample_size)
        model.fit(train_subset)
        score_list.append[get_score(model)]
    return score_list
