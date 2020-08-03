
from .models.model_main_funcs import initialize_model, get_score, fit_model
from .retrieve_data import get_sample


def compute_experimental_result(model_args,
                                train_dataset,
                                test_dataset,
                                sample_size):

    score_list = []
    for i in range(model_args["n_cross_val"]):
        train_subset = get_sample(model_args,
                                  train_dataset,
                                  sample_size,
                                  random_state=i)
        model = initialize_model(model_args, train_subset)
        model = fit_model(model, model_args, train_subset)
        score_list.append(get_score(model, model_args, test_dataset))
    return score_list


