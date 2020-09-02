import os
from pathlib import Path
import glob
import numpy as np
import pickle
import statistics
import matplotlib.pyplot as plt
import seaborn as sns


def plot_result(saving_params):
    """
    This function opens the dictionary containing the stored results
    of previous computation, and display them. Note that it displays
    all results in the dictionary, not only the one that were computed
    during the experiment. It allows to easily compare current experiment
    to previous ones.
    """
    result_dict_path = os.path.join(
        saving_params["result_dict_save_folder"],
        saving_params["result_dict_name"])
    
    with open(result_dict_path, 'rb') as handle:
        result_dict = pickle.load(handle)

    with sns.axes_style("darkgrid"):
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
        number = get_number_of_same_experiments(
            saving_params["plots_save_folder"], saving_params)
        plt.savefig(os.path.join(saving_params["plots_save_folder"],
                                 saving_params["result_dict_name"] 
                                 + str(number+1) + '.png'))


def get_number_of_same_experiments(folder, saving_params):
    return len(glob.glob(folder + "/"
                         + saving_params["result_dict_name"]
                         + "*.png"))

