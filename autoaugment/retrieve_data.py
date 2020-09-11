import numpy as np

from torch.utils.data import Subset
from braindecode.datautil.preprocess import (
    MNEPreproc, NumpyPreproc, preprocess)
from braindecode.datasets import SleepPhysionet
from braindecode.datautil import create_windows_from_events


def get_epochs_data(train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=["scaling", "filtering"], crop_wake_mins=30):

    train_sample = build_epoch(
        train_subjects, recording, crop_wake_mins, preprocessing)
    valid_sample = build_epoch(
        valid_subjects, recording, crop_wake_mins, preprocessing)
    test_sample = build_epoch(test_subjects, recording,
                              crop_wake_mins, preprocessing)

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


def build_epoch(subjects, recording, crop_wake_mins, preprocessing):
    dataset = SleepPhysionet(subject_ids=subjects,
                             recording_ids=recording,
                             crop_wake_mins=crop_wake_mins)

    if preprocessing:
        preprocessors = []
        if "scaling" in preprocessing:
            preprocessors.append[NumpyPreproc(fn=lambda x: x * 1e6)]
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

    return(windows_dataset)
