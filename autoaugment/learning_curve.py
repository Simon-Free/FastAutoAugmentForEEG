import os
from pathlib import Path
import glob
import numpy as np
import pickle
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


def plot_result(saving_params):

    folder = retrieve_folder(saving_params)
    abs_result_dict_path = os.path.join(folder, "result_dict",
                                        saving_params["result_dict_name"])
    Path(abs_result_dict_path).mkdir(parents=True, exist_ok=True)
    with open(abs_result_dict_path, 'rb') as handle:
        result_dict = pickle.load(handle)

    with sns.axes_style("darkgrid"):
        abs_img_path = os.path.join(folder, "plots",
                                    saving_params["result_dict_name"])
        Path(abs_img_path).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", len(result_dict))
        for i in range(len(list(result_dict.keys()))):
            sorted_result = result_dict[list(result_dict.keys())[i]]
            sample_size = list(sorted_result.keys())
            test_mean = [statistics.mean(result_list)
                         for result_list in sorted_result.values()]
            test_std = [statistics.stdev(result_list)
                        for result_list in sorted_result.values()]
            ax.plot(list(sample_size), test_mean,
                    label=list(result_dict.keys())[i], c=clrs[i])
            ax.fill_between(list(sample_size),
                            np.array(test_mean)-np.array(test_std),
                            np.array(test_mean)+np.array(test_std), alpha=0.3,
                            facecolor=clrs[i])
        ax.title.set_text("""
        Diagram representing the accuracy as a function of
            the proportion of the initial data set used""")
        ax.set_xlabel("proportion of the initial training dataset used")
        ax.set_ylabel("model accuracy")
        ax.legend()
        number = get_number_of_same_experiments(abs_img_path, saving_params)
        plt.savefig(folder + "/"
                    + saving_params["result_dict_name"]
                    + str(number+1) + '.png')


def retrieve_folder(saving_params):
    users_with_custom_dir = saving_params["folder"].keys()
    for user in users_with_custom_dir:
        if user in os.getcwd():
            return saving_params["folder"][user]
    default_dir = os.path.join(os.getcwd(), "result_folder")
    Path(default_dir).mkdir(parents=True, exist_ok=True)
    return default_dir


def get_number_of_same_experiments(folder, saving_params):
    return len(glob.glob(folder + "/"
                         + saving_params["result_dict_name"]
                         + "*.png"))
