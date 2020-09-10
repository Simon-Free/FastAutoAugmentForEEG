
from unittest.main import main
import numpy as np
from joblib import Memory

from autoaugment.retrieve_data import get_epochs_data

cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


@memory.cache
def get_dummy_sample():
    train_sample, valid_sample, test_sample = get_epochs_data(
        train_subjects=[0],
        valid_subjects=[1],
        test_subjects=[2],
        recording=[1])
    # for i in range(len(train_sample)):
    #     train_sample[i] = (train_sample[i][0][:50], train_sample[i][1],
    #                        train_sample[i][2])
    # for i in range(len(test_sample)):
    #     test_sample[i] = (test_sample[i][0][:50],
    #                       test_sample[i][1], test_sample[i][2])
    test_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    valid_choice = np.random.choice(
        range(len(test_sample)),
        size=2,
        replace=False)
    train_sample.datasets = [train_sample.datasets[0][350],
                             train_sample.datasets[0][1029],
                             train_sample.datasets[0][1291],
                             train_sample.datasets[0][1650],
                             train_sample.datasets[0][1571]]
    import ipdb
    ipdb.set_trace()
    # train_sample.description = train_sample.description.loc[[0, 1, 2, 3, 4]]
    train_sample.cumulative_sizes = list(range(5))

    test_sample.datasets = [test_sample.datasets[0][test_choice[0]],
                            test_sample.datasets[0][test_choice[1]]]
    valid_sample.datasets = [valid_sample.datasets[valid_choice[0]],
                             valid_sample.datasets[valid_choice[1]]]
    valid_sample.description = valid_sample.description.loc[[0, 1]]
    valid_sample.cumulative_sizes = valid_sample.cumulative_sizes[:2]

    train_sample.transform_list = train_sample.transform_list
    valid_sample.transform_list = valid_sample.transform_list
    test_sample.transform_list = test_sample.transform_list

    return train_sample, valid_sample, test_sample


def take_dataset_subset(windows_dataset, indice_list):
    windows_dataset.windows = windows_dataset.windows[tuple(indice_list)]
    windows_dataset.y = windows_dataset.y[indice_list]
    windows_dataset.crop_inds = windows_dataset.crop_inds[indice_list]


if __name__ == "__main__":
    _, _, _ = get_dummy_sample()
