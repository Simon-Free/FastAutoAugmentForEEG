
import numpy as np
from numpy.core.fromnumeric import var
import torch
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Subset
from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)
from braindecode.datasets import SleepPhysionet
from braindecode.datautil import create_windows_from_events


def get_epochs_data(num_train=None, num_test=None, num_valid=None,
                    train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=["microvolt_scaling", "filtering"], crop_wake_mins=30,
                    random_seed=None):

    if num_train is not None:
        np.random_seed(random_seed)
        rand_sub = np.random.choice(
            tuple(range(83)), size=num_train+num_test+num_valid, replace=True, p=None)
        train_subjects = rand_sub[:num_train]
        test_subjects = rand_sub[num_train:num_train+num_test]
        valid_subjects = rand_sub[num_train+num_test:]

    train_sample = build_epoch(
        train_subjects, recording, crop_wake_mins, preprocessing)
    valid_sample = build_epoch(
        valid_subjects, recording, crop_wake_mins, preprocessing,)
    test_sample = build_epoch(test_subjects, recording,
                              crop_wake_mins, preprocessing)

    return train_sample, valid_sample, test_sample


def get_sample(train_dataset, transform_list, sample_size, random_state=None):
    rng = np.random.RandomState(random_state)
    tf_list_len = len(transform_list)
    len_aug_dataset = len(train_dataset) * tf_list_len
    subset_sample = rng.choice(
        range(int(len_aug_dataset / tf_list_len)),
        size=int(sample_size *
                 len_aug_dataset /
                 tf_list_len),
        replace=False)
    subset_aug_sample = np.array([(np.arange(i*tf_list_len, i*tf_list_len
                                             + tf_list_len))
                                  for i in subset_sample]).flatten()
    subset_aug_labels = np.array([(np.full(tf_list_len, train_dataset[i][1]))
                                  for i in subset_sample]).flatten()

    # train_subset = Subset(
    #     dataset=train_dataset,
    #     indices=subset_aug_sample)
    return subset_aug_sample, subset_aug_labels

    # TODO Trouver autre solution subset.
    # TODO mne-tools/mne-features


def build_epoch(subjects, recording, crop_wake_mins, preprocessing,
                train=True):
    dataset = SleepPhysionet(subject_ids=subjects,
                             recording_ids=recording,
                             crop_wake_mins=crop_wake_mins)

    if preprocessing:
        preprocessors = []
        if "microvolt_scaling" in preprocessing:
            preprocessors.append(NumpyPreproc(fn=lambda x: x * 1e6))
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

    return windows_dataset


def create_label_index_dict(subset_aug_sample, subset_aug_labels):

    list_labels = list(set(subset_aug_labels))
    label_index_dict = {}
    for label in list_labels:
        label_index_dict[label] = []
    for i in range(len(subset_aug_sample)):
        label_index_dict[subset_aug_labels[i]].append(subset_aug_sample[i])
    return(label_index_dict)
