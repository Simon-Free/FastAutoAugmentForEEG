from .compute_model import compute_experimental_result


def main_compute(model_args_list, dataset_args_list, train_dataset,
                 test_dataset, sample_size_list):
    result_dict = {}
    for i in range(len(model_args_list)):
        model_args = model_args_list[i]
        dataset_args = dataset_args_list[i]
        key = (model_args["model_type"] + " + "
               + dataset_args["transform_type"])
        for sample_size in sample_size_list:
            score = compute_experimental_result(model_args,
                                                train_dataset,
                                                test_dataset,
                                                sample_size)
            if key not in result_dict.keys():
                result_dict[key] = {}
            result_dict[key][sample_size] = score
    return(result_dict)
