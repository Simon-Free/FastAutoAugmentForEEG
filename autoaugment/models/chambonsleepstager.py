import torch
from numpy import set_random_seeds
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier

    cuda = torch.cuda.is_available()  # check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = set_random_seeds(seed=model["seed"], cuda=cuda)

n_classes = 5
# Extract number of channels and time steps from dataset
n_channels = train_set[0][0].shape[0]
input_size_samples = train_set[0][0].shape[1]
window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq
n_classes = 5
# Extract number of channels and time steps from dataset
n_channels = train_set[0][0].shape[0]
input_size_samples = train_set[0][0].shape[1]
model = ChambonSleepStager(
    n_channels,
    sfreq,
    n_classes=n_classes,
    input_size_s=input_size_samples / sfreq
)

# Send model to GPU
if cuda:
    model.cuda()


######################################################################
# Training
# --------
#


######################################################################
# We can now train our network. EEGClassifier is a Braindecode object that is
# responsible for managing the training of neural networks. It inherits
# from skorch.NeuralNetClassifier, so the training logic is the same as in
# `Skorch <https://skorch.readthedocs.io/en/stable/>`__.
#


######################################################################
#    **Note**: We use the hyperparameters of Chambon et al. (2018), however
#    these hyperparameters were optimized on a different dataset (MASS SS3) and
#    with a different number of recordings. Therefore, it would make sense to
#    perform hyperparameter optimization if reusing this code on a different
#    dataset.
#


lr = 1e-3
batch_size = 128
n_epochs = 50

train_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=True, name='train_bal_acc',
    lower_is_better=False)
valid_bal_acc = EpochScoring(
    scoring='balanced_accuracy', on_train=True, name='valid_bal_acc',
    lower_is_better=False)
callbacks = [('train_bal_acc', train_bal_acc),
             ('valid_bal_acc', valid_bal_acc)]

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.Adam,
    train_split=predefined_split(valid_set),  # using valid_set for validation
    optimizer__lr=lr,
    batch_size=batch_size,
    callbacks=callbacks,
    device=device
)
# Model training for a specified number of epochs. `y` is None as it is already
# supplied in the dataset.
clf.fit(train_set, y=None, epochs=n_epochs)
