import matplotlib.pyplot as plt


def plot_result(result_dict):

    for key in result_dict.keys():
        sorted_result = {k: v for k, v in sorted(result_dict[key].items(),
                         key=lambda item: item[1])}
        sample_size = sorted_result.keys()
        test_values = sorted_result.values()
        plt.plot(sample_size, test_values, label="key")
    plt.show()


def plot_results(clf):

    # Extract loss and accuracy values for plotting from history object
    results_columns = ['train_loss', 'valid_loss',
                       'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns=results_columns,
                      index=clf.history[:, 'epoch'])

    # get percent of misclass for better visual comparison to loss
    df = df.assign(train_misclass=100 - 100 * df.train_accuracy,
                   valid_misclass=100 - 100 * df.valid_accuracy)

    plt.style.use('seaborn')
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1,
        style=['-', ':'],
        marker='o',
        color='tab:blue',
        legend=False,
        fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel("Misclassification Rate [%]", color='tab:red', fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel("Epoch", fontsize=14)

    # where some data has already been plotted to ax
    handles = []
    handles.append(Line2D([0], [0], color='black', linewidth=1,
                          linestyle='-', label='Train'))
    handles.append(Line2D([0], [0], color='black', linewidth=1,
                          linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()
    plt.savefig('final_hf_vs_dl.png')
    plt.show()