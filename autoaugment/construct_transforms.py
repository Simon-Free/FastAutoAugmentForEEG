import random
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT

from .transforms.signal_merging import merge_two_signals
from .transforms.delaying_signal import delay_signal
from .transforms.noise_addition import add_noise_to_signal
from .transforms.masking import mask_along_time, \
    mask_along_frequency
from .transforms.identity import identity, identity_ml
from .transforms.em_decomposition import merge_two_emd


def construct_transforms(dataset_args, transforms_args):
    transform_list = []
    for transform_name in dataset_args["transform_list"]:
        transform = []
        for operation in transform_name:
            if operation == "merge_two_signals":
                transform.append(TransformSignal(
                    merge_two_signals, transforms_args))
            elif operation == "identity":
                transform.append(TransformSignal(identity, transforms_args))
            elif operation == "identity_ml":
                transform.append(TransformSignal(identity_ml, transforms_args))
            elif operation == "delay_signal":
                transform.append(TransformSignal(
                    delay_signal, transforms_args))
            elif operation == "add_noise_to_signal":
                transform.append(TransformSignal(
                    add_noise_to_signal, transforms_args))
            elif operation == "merge_two_emd":
                transform.append(TransformSignal(
                    merge_two_emd, transforms_args))
            elif operation == "randaugment":
                transform.append(TransformSignal(
                    construct_randaugment, transforms_args))
            elif operation == "mask_along_time":
                transform.append(TransformFFT(
                    mask_along_time, transforms_args))
            elif operation == "mask_along_frequency":
                transform.append(TransformFFT(
                    mask_along_frequency, transforms_args))
            else:
                raise ValueError(
                    "This transform is currently not implemented")
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
        datum = transf.transform(datum)
    return(datum)
