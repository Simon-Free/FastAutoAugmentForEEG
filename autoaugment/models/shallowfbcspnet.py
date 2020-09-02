import torch
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode.models import ShallowFBCSPNet
from braindecode.util import set_random_seeds
from braindecode import EEGClassifier


def get_shallowfbcspnet(model_args):

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    seed = model_args["seed"]
    # = 20200220  random seed to make results reproducible
    # Set random seed to be able to reproduce results
    if seed:
        set_random_seeds(seed=seed, cuda=cuda)

    model = ShallowFBCSPNet(
        model_args["n_chans"],
        model_args["n_classes"]+1,
        input_window_samples=model_args["input_window_samples"],
        final_conv_length='auto',
        )
    if cuda:
        model.cuda()
    
    lr = model_args["lr"]  # 0.0625 * 0.01
    weight_decay = model_args["weight_decay"]  # 0

    # For deep4 they should be:
    # lr = 1 * 0.01
    # weight_decay = 0.5 * 0.001
    n_epochs = model_args["n_epochs"]  # = 100
    batch_size = model_args["batch_size"]  # = F
    if model_args["train_split"] is None:
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            # using test_sample for validation
            train_split=None,
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy", ("lr_scheduler",
                            LRScheduler('CosineAnnealingLR',
                                        T_max=n_epochs - 1)),
            ],
            device=device,
        )  # torch.in torch.out

    else:
        clf = EEGClassifier(
            model,
            criterion=torch.nn.NLLLoss,
            optimizer=torch.optim.AdamW,
            # using test_sample for validation
            optimizer__lr=lr,
            optimizer__weight_decay=weight_decay,
            batch_size=batch_size,
            callbacks=[
                "accuracy", 
                ("lr_scheduler",
                 LRScheduler(
                    'CosineAnnealingLR',
                    T_max=n_epochs - 1)),
            ],
            device=device,
        )  # torch.in torch.out

    return clf
