import numpy as np


def merge_two_signals(datum, params):

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
