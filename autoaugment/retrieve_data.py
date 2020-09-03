import numpy as np
import mne

from mne.datasets.sleep_physionet.age import fetch_data
from torch.utils.data import Subset

from braindecode.datasets import create_from_mne_epochs

from joblib import Memory

cachedir = 'cache_dir'
memory = Memory(cachedir, verbose=0)


# XXX use tuple not list
def get_epochs_data(train_subjects=list(range(15)),
                    test_subjects=list(range(15, 20)), recording=[1, 2],
                    dummy=False):
    train_files_list = fetch_data(subjects=train_subjects, recording=recording)
    test_files_list = fetch_data(subjects=test_subjects, recording=recording)
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
                                  baseline=None).load_data()
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
                                 event_id=event_id,
                                 tmin=0., tmax=tmax, baseline=None).load_data()

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
    subset_aug_sample = np.array(
        [np.array([i * tf_list_len + j for j in range(tf_list_len)])
         # XXX use arange
         for i in subset_sample]).flatten()
    train_subset = Subset(
       dataset=train_dataset,
       indices=subset_aug_sample)
    return train_subset


# XXX : move to test folder
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

    # TODO Assert getitem sur GPU.
    # TODO Trouver autre solution subset.
    # TODO np.array(list(dataset))
    # TODO mne-tools/mne-features
