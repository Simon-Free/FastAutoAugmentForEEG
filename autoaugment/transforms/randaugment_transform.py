import random

from braindecode.datasets.transform_classes import TransformFFT, TransformSignal

from .signal_merging import merge_two_signals
from .delaying_signal import delay_signal
from .noise_addition import add_noise_to_signal
from .masking import mask_along_axis_random
from .identity import identity
from .emd_decomposition import merge_two_imfs


# TODO : régler le problème du delay
def rand_transf(datum, params):
    n_transf = params["n_transf"]
    chosen_transf = []
    for i in range(n_transf):
        new_transf = random.choice(params["transform_list"])
        if new_transf == "merge_two_signals":
            chosen_transf.append(TransformSignal(merge_two_signals))
        elif new_transf == "identity":
            chosen_transf.append(TransformSignal(identity))
        elif new_transf == "delay_signal":
            chosen_transf.append(TransformSignal(delay_signal))
        elif new_transf == "add_noise_to_signal":
            chosen_transf.append(TransformSignal(add_noise_to_signal))
        elif new_transf == "emd_decomp"
