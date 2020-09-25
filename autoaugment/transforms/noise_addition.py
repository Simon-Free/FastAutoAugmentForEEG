import numpy as np
import random
import torch


def add_noise_to_signal(datum, params):

    signal = datum.X
    scale = torch.mean(torch.abs(signal)).item()
    noise = torch.Tensor(np.random.normal(
        loc=0.0, scale=scale, size=signal.shape))
    signal = signal + params["magnitude"]*0.1*noise
    datum.X = signal
    return datum


def add_noise_to_signal_with_proba(datum, params):

    rand = random.random()
    if rand <= 0.5:
        return(datum)
    else:
        return(add_noise_to_signal(datum, params))


def add_noise_to_signal_only_one_signal(datum, params):

    signal = datum.X
    scale = torch.mean(torch.abs(signal)).item()

    noise = torch.Tensor(np.random.normal(
        loc=0.0, scale=scale, size=signal.shape[1]))
    rand = random.random()
    if rand <= 0.5:
        signal[0, :] = signal[0, :] + params["magnitude"]*0.1*noise
    else:
        signal[1, :] = signal[1, :] + params["magnitude"]*0.1*noise
    datum.X = signal
    return datum
