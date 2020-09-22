import random
import numpy as np
from PyEMD import EMD
import torch


def merge_two_emd(datum, params):
    signal = datum.X
    y = datum.y
    train_sample = params["train_sample"]
    label_index_dict = params["label_index_dict"]
    other_signal_index = random.choice(label_index_dict[y])
    other_signal = train_sample[other_signal_index][0]
    final_signal = np.zeros(signal.shape)
    for i in range(signal.shape[0]):
        imfs_signal = get_same_shaped_imfs(signal[i])
        other_imfs_signal = get_same_shaped_imfs(other_signal[i])
        for j in range(12):
            if random.random() <= params["magnitude"]:
                final_signal[i] += other_imfs_signal[j]
            else:
                final_signal[i] += imfs_signal[j]
    datum.X = torch.Tensor(final_signal)
    return datum


def get_same_shaped_imfs(signal):
    emd = EMD()
    try:
        imfs_signal = emd.emd(signal.numpy(), max_imf=12)
    except AttributeError:
        imfs_signal = emd.emd(signal, max_imf=12)
    if imfs_signal.shape[0] > 12:
        imfs_signal = imfs_signal[:12, :]
    new_imfs = np.zeros((12, signal.shape[0]))
    new_imfs[:imfs_signal.shape[0], :] = imfs_signal
    return new_imfs
