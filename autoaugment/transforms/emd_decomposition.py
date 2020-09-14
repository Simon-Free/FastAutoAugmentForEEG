import random
import numpy as np
from PyEMD import EMD


def merge_two_emfs(datum, params):
    signal = datum.X
    y = datum.y
    train_sample = params["train_sample"]
    label_index_dict = params["label_index_dict"]
    other_signal_index = random.choice(label_index_dict[y])
    other_signal = train_sample[other_signal_index][1]
    imfs_signal = get_same_shaped_imfs(signal)
    other_imfs_signal = get_same_shaped_imfs(other_signal)
    final_signal = np.zeros(imfs_signal.shape[1])
    for i in range(12):
        if random.random() <= params["magnitude"]:
            final_signal += other_imfs_signal[i, :]
        else:
            final_signal += imfs_signal[i, :]
    datum.X = final_signal
    return datum


def get_same_shaped_imfs(signal):
    emd = EMD(max_imfs=12)
    imfs_signal = emd(signal)
    new_imfs = np.zeros((imfs_signal.shape[1], 12))
    new_imfs[:imfs_signal.shape[0], :] = imfs_signal
    return new_imfs
