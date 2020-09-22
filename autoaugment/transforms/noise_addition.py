import numpy as np
import torch


def add_noise_to_signal(datum, params):

    signal = datum.X
    # scale = torch.mean(torch.abs(signal)).item()
    scale = 50
    noise = torch.Tensor(np.random.normal(
        loc=0.0, scale=scale, size=signal.shape))
    signal += params["magnitude"]*noise
    return datum
