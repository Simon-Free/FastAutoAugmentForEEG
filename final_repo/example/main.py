from ..autoaugment.learning_curve import plot_result
from ..autoaugment.compute_model import compute_experimental_result


def main_compute():
    result_dict = {}
    for i in range(len(model_args_list)):
        model_args = model_args_list[i]
        dataset_args = dataset_args_list[i]
        key = (model_args["model_name"] + " + "
               + dataset_args["dataset_name"])
        sample_size = dataset_args["sample_size"]
        score = compute_experimental_result(model_args, dataset_args)
        if key not in result_dict.keys():
            result_dict[key] = {}
        result_dict[key][sample_size] = score
    plot_result(result_dict)
