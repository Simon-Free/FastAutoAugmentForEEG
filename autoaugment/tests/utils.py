
import numpy as np
from joblib import Memory

from autoaugment.retrieve_data import get_epochs_data

cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


@memory.cache
def get_dummy_sample():
    train_sample, test_sample = get_epochs_data(
        train_subjects=[0],
        test_subjects=[1],
        recording=[1], dummy=True)
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
    train_sample.datasets = [train_sample.datasets[350],
                             train_sample.datasets[1029],
                             train_sample.datasets[1291],
                             train_sample.datasets[1650],
                             train_sample.datasets[1571]]
    train_sample.description = train_sample.description.loc[[0, 1, 2, 3, 4]]
    train_sample.cumulative_sizes = train_sample.cumulative_sizes[:5]

    test_sample.datasets = [test_sample.datasets[test_choice[0]],
                            test_sample.datasets[test_choice[1]]]
    test_sample.description = test_sample.description.loc[[0, 1, 2, 3, 4]]
    test_sample.cumulative_sizes = test_sample.cumulative_sizes[:2]
    # sub_train_sample = Subset(train_sample, [350, 1029, 1291, 1650, 1571])
    train_sample.transform_list = train_sample.transform_list
    # sub_test_sample = Subset(test_sample, test_choice)
    test_sample.transform_list = test_sample.transform_list

    return train_sample, test_sample
