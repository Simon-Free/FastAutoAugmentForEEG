import matplotlib.pyplot as plt
import numpy as np
import statistics
import seaborn as sns


def plot_result(result_dict, save_name):
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots()
        clrs = sns.color_palette("husl", len(result_dict))
        import ipdb; ipdb.set_trace()
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
        plt.savefig(save_name + '.png')
