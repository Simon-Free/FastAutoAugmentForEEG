import numpy as np
import mne

from mne.datasets.sleep_physionet.age import fetch_data
from torch.utils.data import Subset

from braindecode.datasets import create_from_mne_epochs


def get_epochs_data(train_subjects=tuple(range(0, 60)),
                    test_subjects=tuple(range(60, 83)), recording=[1, 2],
                    dummy=False):
    train_files_list = fetch_data(
        subjects=train_subjects, recording=recording, on_missing="ignore")
    test_files_list = fetch_data(
        subjects=test_subjects, recording=recording, on_missing="ignore")
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

    epochs_train_list = []
    epochs_test_list = []

    for subj_files in train_files_list:
        raw_train = mne.io.read_raw_edf(subj_files[0])
        annot_train = mne.read_annotations(subj_files[1])
        raw_train.set_annotations(annot_train, emit_warning=False)
        raw_train.set_channel_types(mapping)
        events_train, _ = mne.events_from_annotations(
            raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        tmax = 30. - 1. / raw_train.info['sfreq']
        epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                                  event_id=event_id, tmin=0., tmax=tmax,
                                  baseline=None, on_missing='warn').load_data()
        epochs_train.drop_bad()
        epochs_train_list.append(epochs_train)

    for subj_files in test_files_list:
        raw_test = mne.io.read_raw_edf(subj_files[0])
        annot_test = mne.read_annotations(subj_files[1])
        raw_test.set_annotations(annot_test, emit_warning=False)
        raw_test.set_channel_types(mapping)
        events_test, _ = mne.events_from_annotations(
            raw_test, event_id=annotation_desc_2_event_id, chunk_duration=30.)
        epochs_test = mne.Epochs(raw=raw_test, events=events_test,
                                 event_id=event_id, tmin=0., tmax=tmax,
                                 baseline=None, on_missing='warn').load_data()

        epochs_test.drop_bad()
        epochs_test_list.append(epochs_test)

    if dummy:
        train_sample = create_from_mne_epochs(
            epochs_train_list,
            window_size_samples=3000,
            window_stride_samples=3000,
            drop_last_window=False)
        # import ipdb; ipdb.set_trace()
    #    train_sample.cumulative_sizes()

        test_sample = create_from_mne_epochs(
            epochs_test_list, window_size_samples=3000,
            window_stride_samples=3000,
            drop_last_window=False)

    else:
        train_sample = create_from_mne_epochs(
            epochs_train_list,
            window_size_samples=3000,
            window_stride_samples=3000,
            drop_last_window=False)
        # import ipdb; ipdb.set_trace()
    #    train_sample.cumulative_sizes()

        test_sample = create_from_mne_epochs(
            epochs_test_list, window_size_samples=3000,
            window_stride_samples=3000,
            drop_last_window=False)

    return train_sample, test_sample


def get_sample(train_dataset, sample_size, random_state=None):
    rng = np.random.RandomState(random_state)
    tf_list_len = len(train_dataset.transform_list)
    subset_sample = rng.choice(
        range(int(len(train_dataset) / len(train_dataset.transform_list))),
        size=int(sample_size *
                 len(train_dataset) /
                 len(train_dataset.transform_list)),
        replace=False)
    subset_aug_sample = np.array([np.arange(i*tf_list_len, i*tf_list_len + tf_list_len)
                                  for i in subset_sample]).flatten()
    train_subset = Subset(
        dataset=train_dataset,
        indices=subset_aug_sample)
    return train_subset

    # TODO Assert getitem sur GPU.
    # TODO Trouver autre solution subset.
    # TODO np.array(list(dataset))
    # TODO mne-tools/mne-features
