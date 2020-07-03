import matplotlib.pyplot as plt


def plot_result(result_dict):

    for key in result_dict.keys():
        sorted_result = {k: v for k, v in sorted(result_dict[key].items(),
                         key=lambda item: item[1])}
        sample_size = sorted_result.keys()
        test_values = sorted_result.values()
        plt.plot(sample_size, test_values, label="key")
    plt.show()



