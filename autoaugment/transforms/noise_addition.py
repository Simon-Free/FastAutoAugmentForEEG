import numpy as np


def merge_two_signals(signal, y, params):

    noise = np.random.normal(
        loc=0.0, scale=np.mean(np.abs(signal)), size=None)
    final_signal = (1 - params["magnitude"])*signal + params["magnitude"]*noise
    return final_signal
