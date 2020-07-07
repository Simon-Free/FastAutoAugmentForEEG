import cProfile
import pstats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.time_frequency import psd_welch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split

from braindecode.util import set_random_seeds
from braindecode.models import ShallowFBCSPNet
from braindecode import EEGClassifier
from braindecode.datasets import create_from_mne_epochs

# insert at 1, 0 is the script path (or '' in REPL)
mne.set_log_level("warning")


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
    return epochs_train_list, epochs_test_list
    # TODO : comment braindecode modifie les annotations ? Comment les 
    # modifier nous-même éventuellement ?


def ml_custom_preprocessing(epochs_train_list, epochs_test_list):
    final_epochs_train = mne.concatenate_epochs(epochs_train_list)
    final_epochs_test = mne.concatenate_epochs(epochs_test_list)
    
    pipe = make_pipeline(FunctionTransformer(eeg_power_band, validate=False),
                         RandomForestClassifier(n_estimators=100,
                                                random_state=42))
    y_train = final_epochs_train.events[:, 2]
    y_test = final_epochs_test.events[:, 2]
    return pipe, final_epochs_train, y_train, y_test, final_epochs_test


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=30.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


def ml_train_and_get_accuracy(pipe, epochs_train, y_train,
                              y_test, epochs_test):

    pipe.fit(epochs_train, y_train)
    # Test
    y_pred = pipe.predict(epochs_test)
    # Assess the results
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: {}".format(acc))


def dl_custom_preprocessing(epochs_train_list, epochs_test_list):
    train_sample = create_from_mne_epochs(
        epochs_train_list,
        window_size_samples=3000,
        window_stride_samples=3000,
        drop_last_window=False)

    test_sample = create_from_mne_epochs(
        epochs_test_list, window_size_samples=3000,
        window_stride_samples=3000,
        drop_last_window=False)
    # TODO : enquêter sur pourquoi ça prend 20 minutes (conversion de données, float64 -> float32, charger les données en mémoire)
    # __getitem__ renvoie triplet (données, label, [epochs_idx, debut, fin])
    # import ipdb; ipdb.set_trace()
    return train_sample, test_sample


def define_model(train_sample, test_sample):
    # check if GPU is available, if True chooses to use it
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = 20200220  # random seed to make results reproducible
    # Set random seed to be able to reproduce results
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 6
    # Extract number of chans and time steps from dataset
    n_chans = train_sample[0][0].shape[0]
    input_window_samples = train_sample[0][0].shape[1]

    model = ShallowFBCSPNet(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        final_conv_length='auto',
    )

    # Send model to GPU
    if cuda:
        model.cuda()


# These values we found good for shallow network:
    lr = 0.0625 * 0.01
    weight_decay = 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001
    n_epochs = 100
    batch_size = 64

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        # using test_sample for validation
        train_split=predefined_split(test_sample),
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler",
                         LRScheduler('CosineAnnealingLR',
                                     T_max=n_epochs - 1)),
        ],
        device=device,
    )
    return clf, n_epochs
# Model training for a specified number of epochs.
# `y` is None as it is already supplied
# in the dataset.


def train_model(clf, train_sample, n_epochs):

    clf.fit(train_sample, y=None, epochs=n_epochs)
    return(clf)


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


def main_ml(epochs_train, epochs_test):
    pipe, epochs_train, y_train, y_test, epochs_test = \
        ml_custom_preprocessing(epochs_train, epochs_test)
    ml_train_and_get_accuracy(pipe, epochs_train, y_train, y_test, epochs_test)


def main_dl(epochs_train, epochs_test):
    train_sample, test_sample = dl_custom_preprocessing(epochs_train, epochs_test)
    clf, n_epochs = define_model(train_sample, test_sample)
    clf = train_model(clf, train_sample, n_epochs)
    plot_results(clf)


if __name__ == "__main__":
    epochs_train_list, epochs_test_list = get_epochs_data()
    cProfile.run("main_ml(epochs_train_list, epochs_test_list)", "ml_results")
    cProfile.run("main_dl(epochs_train_list, epochs_test_list)", "dl_results")
    ml_stats = pstats.Stats("ml_results")
    dl_stats = pstats.Stats("dl_results")
    ml_stats.print_stats("ml_vs_dl.py")
    dl_stats.print_stats("ml_vs_dl.py")
