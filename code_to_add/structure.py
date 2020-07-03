# Transformer un dataset en truc utilisable
# La fonction c'est
import matplotlib.pyplot as plt
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


def plot_result(model_args_list, dataset_args_list):
    result_dict = {}
    for i in range(len(model_args_list)):
        model_args = model_args_list[i]
        dataset_args = dataset_args_list[i]
        key = (model_args["model_name"] + " + " 
               +  dataset_args["dataset_name"])
        sample_size = dataset_args["sample_size"]
        score = compute_experimental_result(model_args, dataset_args)
        if key not in result_dict.keys():
            result_dict[key] = {}
        result_dict[key][sample_size] = score

    for key in result_dict.keys():
        sorted_result = {k: v for k, v in sorted(result_dict[key].items(), key=lambda item: item[1])}
        sample_size = sorted_result.keys()
        test_values = sorted_result.values()
        plt.plot(sample_size, test_values, label="key") # plotting t, a separately 
    plt.show()
