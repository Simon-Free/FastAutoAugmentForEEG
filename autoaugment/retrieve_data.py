

import numpy as np
from numpy.core.fromnumeric import var

import torch
from torch.utils.data import Subset
from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)
from braindecode.datasets import SleepPhysionet
from braindecode.datautil import create_windows_from_events
from torch.utils.data import dataset


def get_epochs_data(train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=["microvolt_scaling", "filtering"], crop_wake_mins=30):

    train_sample, std_train, avg_train = build_epoch(
        train_subjects, recording, crop_wake_mins, preprocessing)
    valid_sample, _, _ = build_epoch(
        valid_subjects, recording, crop_wake_mins, preprocessing,
        std_train, avg_train)
    test_sample, _, _ = build_epoch(test_subjects, recording,
                                    crop_wake_mins, preprocessing,
                                    std_train, avg_train)

    return train_sample, valid_sample, test_sample


def get_sample(train_dataset, sample_size, random_state=None):
    rng = np.random.RandomState(random_state)
    tf_list_len = len(train_dataset.transform_list)
    subset_sample = rng.choice(
        range(int(len(train_dataset) / len(train_dataset.transform_list))),
        size=int(sample_size *
                 len(train_dataset) /
                 len(train_dataset.transform_list)),
        replace=False)
    subset_aug_sample = np.array([np.arange(i*tf_list_len, i*tf_list_len
                                            + tf_list_len)
                                  for i in subset_sample]).flatten()
    train_subset = Subset(
        dataset=train_dataset,
        indices=subset_aug_sample)
    return train_subset

    # TODO Trouver autre solution subset.
    # TODO mne-tools/mne-features


def build_epoch(subjects, recording, crop_wake_mins, preprocessing,
                std_train=None,
                avg_train=None):
    dataset = SleepPhysionet(subject_ids=subjects,
                             recording_ids=recording,
                             crop_wake_mins=crop_wake_mins)

    if preprocessing:
        preprocessors = []
        if "microvolt_scaling" in preprocessing:
            preprocessors.append(NumpyPreproc(fn=lambda x: x * 1e6))
        elif "standard_scaling" in preprocessing:
            if std_train is None:
                std, avg = compute_stats_of_train_algorithm(dataset)
                preprocessors.append(NumpyPreproc(fn=lambda x: (x - avg)/std))
            else:
                preprocessors.append(NumpyPreproc(
                    fn=lambda x: (x - avg_train)/std_train))

        if "filtering" in preprocessing:
            high_cut_hz = 30
            preprocessors.append(
                MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz)
            )
        # Transform the data
        preprocess(dataset, preprocessors)
    mapping = {  # We merge stages 3 and 4 following AASM standards.
        'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3': 3,
        'Sleep stage 4': 3,
        'Sleep stage R': 4
    }

    window_size_s = 30
    sfreq = 100
    window_size_samples = window_size_s * sfreq

    windows_dataset = create_windows_from_events(
        dataset, trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples, preload=True,
        mapping=mapping)

    return windows_dataset, std_train, avg_train


def create_label_index_dict(train_dataset):
    y = [train_dataset[i][1] for i in range(len(train_dataset))]
    list_labels = list(set(y))
    label_index_dict = {}
    for label in list_labels:
        label_index_dict[label] = []
    for i in range(len(y)):
        label_index_dict[y[i]].append(i)
    return(label_index_dict)


def compute_stats_of_train_algorithm(train_sample):
    avg, n_samples, M2 = 0, 0, 0
    for i in range(len(train_sample)):
        X = train_sample[i][0]
        new_avg = torch.mean(X)
        new_n_samples = X.shape[0]*X.shape[1]
        new_M2 = (new_n_samples - 1)*torch.var(X)
        M2 = parallel_variance(n_samples, avg, M2,
                               new_n_samples, new_avg, new_M2)
        avg = (n_samples * avg + new_n_samples * new_avg) / \
            (n_samples + new_n_samples)
        n_samples += new_n_samples
    var = M2 / (n_samples - 1)
    return avg, np.sqrt(var)


def parallel_variance(n_a, avg_a, M2_a, n_b, avg_b, M2_b):
    """ source
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    n = n_a + n_b
    delta = avg_b - avg_a
    M2 = M2_a + M2_b + delta ** 2 * n_a * n_b / n
    return M2
