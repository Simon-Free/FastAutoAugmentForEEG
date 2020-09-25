import random
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT

from .transforms.signal_merging import merge_two_signals
from .transforms.delaying_signal import delay_signal
from .transforms.noise_addition import add_noise_to_signal, \
    add_noise_to_signal_only_one_signal, add_noise_to_signal_with_proba
from .transforms.masking import mask_along_time, \
    mask_along_frequency
from .transforms.identity import identity, identity_ml
from .transforms.em_decomposition import merge_two_emd


def construct_transforms(dataset_args, transforms_args):
    transforms_dict = {
        "merge_two_signals": TransformSignal(
            merge_two_signals, transforms_args),
        "identity": TransformSignal(
            identity, transforms_args),
        "identity_ml": TransformSignal(
            identity_ml, transforms_args),
        "delay_signal": TransformSignal(
            delay_signal, transforms_args),
        "add_noise_to_signal": TransformSignal(
            add_noise_to_signal, transforms_args),
        "add_noise_to_signal_only_one_signal": TransformSignal(
            add_noise_to_signal_only_one_signal, transforms_args),
        "add_noise_to_signal_with_proba": TransformSignal(
            add_noise_to_signal_with_proba, transforms_args),
        "merge_two_emd": TransformSignal(
            merge_two_emd, transforms_args),
        "randaugment": TransformSignal(
            construct_randaugment, transforms_args),
        "mask_along_time": TransformFFT(
            mask_along_time, transforms_args),
        "mask_along_frequency": TransformFFT(
            mask_along_frequency, transforms_args),
    }
    transform_list = []
    for transform_name in dataset_args["transform_list"]:
        transform = []
        for operation in transform_name:
            transform.append(transforms_dict[operation])
        transform_list.append(transform)
    return transform_list


def construct_randaugment(datum, params):

    n_transf = params["n_transf"]
    chosen_transf = random.choices(
        params["randaugment_transform_list"], k=n_transf)
    chosen_transf.append("identity")
    wrapped_chosen_transf = {"transform_list": [chosen_transf]}
    constructed_chosen_transf = construct_transforms(
        wrapped_chosen_transf, params)
    for transf in constructed_chosen_transf[0]:
        datum = transf(datum)
    return(datum)
