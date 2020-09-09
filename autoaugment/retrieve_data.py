import numpy as np
import mne

from torch.utils.data import Subset
from mne.datasets.sleep_physionet.age import fetch_data
from braindecode.datautil import NumpyPreproc, MNEPreproc, preprocess
from braindecode.datasets import create_from_mne_epochs


def get_epochs_data(train_subjects=tuple(range(0, 50)),
                    valid_subjects=tuple(range(50, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    preprocessing=True):

    train_sample = build_epoch(train_subjects, recording, preprocessing)
    valid_sample = build_epoch(valid_subjects, recording, preprocessing)
    test = build_epoch(test_subjects, recording, preprocessing)
                               
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


def build_epoch(subjects, recording, preprocessing):
    files_list = fetch_data(
        subjects=subjects, recording=recording, on_missing="ignore")
    mapping = {'EOG horizontal': 'eog',
               'Resp oro-nasal': 'misc',
               'EMG submental': 'misc',
               'Temp rectal': 'misc',
               'Event marker': 'misc'}
    annotation_desc_2_event_id = {'Sleep stage W': 1,
                                  'Sleep stage 1': 2,
                                  'Sleep stage 2': 3,
                                  'Sleep stage 3': 4,
                                  'Sleep stage 4': 4,
                                  'Sleep stage R': 5}
    event_id = {'Sleep stage W': 1,
                'Sleep stage 1': 2,
                'Sleep stage 2': 3,
                'Sleep stage 3/4': 4,
                'Sleep stage R': 5}

    epochs_list = []
    for subj_file in files_list:
        raw = mne.io.read_raw_edf(subj_file[0])
        annot = mne.read_annotations(subj_file[1])
        raw.set_annotations(annot, emit_warning=False)
        raw.set_channel_types(mapping)
        tmax = 30. - 1. / raw.info['sfreq']
        events, _ = mne.events_from_annotations(
            raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        epochs = mne.Epochs(raw=raw, events=events,
                            event_id=event_id, tmin=0., tmax=tmax,
                            baseline=None, on_missing='warn').load_data()
        epochs.drop_bad()
        epochs_list.append(epochs)

    dataset = create_from_mne_epochs(
        epochs_list,
        window_size_samples=3000,
        window_stride_samples=3000,
        drop_last_window=False)

    if preprocessing:
        high_cut_hz = 30
        preprocessors = [
            # convert from volt to microvolt,
            # directly modifying the numpy array
            NumpyPreproc(fn=lambda x: x * 1e6),
            # bandpass filter
            MNEPreproc(fn='filter', l_freq=None, h_freq=high_cut_hz),
        ]
        # Transform the data
        preprocess(dataset, preprocessors)
    return(dataset)
