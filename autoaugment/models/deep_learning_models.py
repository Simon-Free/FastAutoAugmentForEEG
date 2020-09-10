import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.models import ShallowFBCSPNet, SleepStager
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier


def get_deep_learning_model(model_args, valid_dataset):
    cuda = torch.cuda.is_available()
    device = model_args["device"] if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = model_args["seed"]
    # = 20200220  random seed to make results reproducible
    # Set random seed to be able to reproduce results
    if seed:
        set_random_seeds(seed=seed, cuda=cuda)

    if model_args["model_name"] == "ShallowFBCSPNET":
        model = ShallowFBCSPNet(
            model_args["n_chans"],
            model_args["n_classes"]+1,
            input_window_samples=model_args["input_window_samples"],
            final_conv_length='auto',
        )
    elif model_args["model_name"] == "SleepStager":
        model = model = SleepStager(
            n_channels=model_args["n_chans"],
            sfreq=model_args["sfreq"],
            n_classes=model_args["n_classes"],
            input_size_s=model_args["input_window_samples"] /
            model_args["sfreq"]
        )
    else:
        raise ValueError("Boom !")

    if cuda:
        model.cuda()

    clf = EEGClassifier(
        model,
        criterion=model_args["criterion"],
        optimizer=torch.optim.AdamW,
        # using test_sample for validation
        train_split=predefined_split(valid_dataset),
        optimizer__lr=model_args["lr"],
        optimizer__weight_decay=model_args["weight_decay"],
        batch_size=model_args["batch_size"],
        callbacks=[
            "accuracy", ("lr_scheduler",
                         LRScheduler('CosineAnnealingLR',
                                     T_max=model_args["n_epochs"] - 1)),
        ],
        device=device,
    )  # torch.in torch.out

    return clf
