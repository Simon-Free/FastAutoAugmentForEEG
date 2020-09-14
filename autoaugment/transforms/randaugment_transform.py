import random

from braindecode.datasets.transform_classes import TransformFFT, \
    TransformSignal

from .signal_merging import merge_two_signals
from .delaying_signal import delay_signal
from .noise_addition import add_noise_to_signal
from .masking import mask_along_axis_random
from .identity import identity
from .em_decomposition import merge_two_imfs


# TODO : régler le problème du delay
def rand_transf(datum, params):
    n_transf = params["n_transf"]
    chosen_transf = []
    for i in range(n_transf):
        new_transf = random.choice(params["transform_list"])
        if new_transf == "merge_two_signals":
            chosen_transf.append(TransformSignal(merge_two_signals, params))
        elif new_transf == "identity":
            chosen_transf.append(TransformSignal(identity, params))
        elif new_transf == "delay_signal":
            chosen_transf.append(TransformSignal(delay_signal, params))
        elif new_transf == "add_noise_to_signal":
            chosen_transf.append(TransformSignal(add_noise_to_signal, params))
        elif new_transf == "merge_two_imfs":
            chosen_transf.append(TransformSignal(merge_two_imfs, params))
        elif new_transf == "mask_along_axis_random":
            chosen_transf.append(TransformFFT(mask_along_axis_random, params))
    chosen_transf.append(TransformSignal(identity, params))
    for transf in chosen_transf:
        datum = transf.transform(datum)
    return(datum)
