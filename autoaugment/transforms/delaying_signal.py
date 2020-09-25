import numpy as np
import torch


def delay_signal(datum, params):

    signal = datum.X
    new_signal = torch.zeros(signal.shape)
    value_magnitude = int(np.round(params["magnitude"] * signal.shape[1]))
    if value_magnitude != 0:
        new_signal[:, :value_magnitude] = signal[:, -value_magnitude:]
        new_signal[:, value_magnitude:] = signal[:, :-value_magnitude]
    datum.X = new_signal

    return datum


# Not Working when subsampling, lots of problem ...
def delay_signal_complicated(datum, params):

    signal = datum.X
    index = datum.index
    train_sample = params["train_sample"]
    if index != 0:
        previous_signal = train_sample[index - 1][0]
        new_signal = np.zeros(signal.shape)
        value_magnitude = np.round(params["magnitude"] * signal.shape)
        new_signal[:value_magnitude] = previous_signal[-value_magnitude:]
        new_signal[value_magnitude:] = signal[:-value_magnitude]
        datum.X = new_signal

    return datum.X
