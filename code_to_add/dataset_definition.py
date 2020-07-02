import matplotlib.pyplot as plt
cachedir = 'cache_dir'
from joblib import Memory
from braindecode.datasets import WindowsDataset
memory = Memory(cachedir, verbose=0)

@memory.cache
def get_epochs_data(train_subjects=list(range(15)), test_subjects=list(range(15, 20))):
    ALICE, BOB = 0, 1
#     [alice_files, bob_files] = fetch_data(subjects=[ALICE, BOB], recording=[1])
    train_files_list = fetch_data(subjects=train_subjects)
    test_files_list = fetch_data(subjects=test_subjects)
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
                                  baseline=None)
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
                                 tmin=0., tmax=tmax, baseline=None)
        epochs_test.drop_bad()
        epochs_test_list.append(epochs_test)

    train_sample = create_from_mne_epochs(
        epochs_train_list,
        window_size_samples=3000,
        window_stride_samples=3000,
        drop_last_window=False)
    test_sample = create_from_mne_epochs(
        epochs_test_list, window_size_samples=3000,
        window_stride_samples=3000,
        drop_last_window=False)

    return train_sample, test_sample

class TransformDataset(WindowsDataset):
    def __init__(self, windows, description=None, transform_list=[lambda x: x]):
        super().__init__(self, windows, description=None)
        self.transform_list = transform_list
        self.len_tf_list = len(transform_list)
    def __getitem__(self, index):
        X = self.windows.get_data(item=index)[0].astype('float32')
        y = self.y[index]
        img_index = index // self.len_tf_list
        tf_index = index % self.len_tf_list
        X = self.transform_list[tf_index](X)
        # necessary to cast as list to get list of
        # three tensors from batch, otherwise get single 2d-tensor...
        crop_inds = list(self.crop_inds[index])
        return X, y, crop_inds
    def __len__(self):
        return len(self.windows.events)*len(self.transform_list)

