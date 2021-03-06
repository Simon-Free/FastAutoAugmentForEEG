from operator import add
from numpy.testing import assert_array_equal, \
    assert_array_almost_equal
from testfixtures import compare
import random

import numpy as np
import torch
from numpy.testing import assert_almost_equal

from braindecode.datasets.base import Datum
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.transforms.identity import identity
from autoaugment.tests.utils import get_dummy_sample
from autoaugment.transforms.masking import mask_along_axis, \
    mask_along_time
from autoaugment.retrieve_data import create_label_index_dict, get_sample
from autoaugment.construct_transforms import construct_transforms
from autoaugment.config import transforms_args
from autoaugment.transforms.em_decomposition import merge_two_emd
from autoaugment.transforms.delaying_signal import delay_signal
from autoaugment.transforms.signal_merging import merge_two_signals
from autoaugment.transforms.noise_addition import add_noise_to_signal, \
    add_noise_to_signal_with_proba, add_noise_to_signal_only_one_signal
from autoaugment.transforms.randaugment_transform import randaugment


def test_mask_along_axis_nonrandom():

    params_masking_nonrandom_test_1 = {
        "mask_value": 0.0,
        "mask_start": 5,
        "mask_end": 10,
        "axis": 2,
    }

    params_masking_nonrandom_test_2 = {
        "mask_value": 0.0,
        "mask_start": 4,
        "mask_end": 25,
        "axis": 1,
    }

    train_sample, _, _ = get_dummy_sample()

    datum = Datum(X=train_sample[0][0], y=train_sample[0][1])
    datum_with_tr = Datum(X=train_sample[0][0], y=train_sample[0][1])

    datum_spec = TransformFFT(identity)(datum)
    datum_spec_with_tr = TransformFFT(
        mask_along_axis, params_masking_nonrandom_test_1
    )(datum_with_tr)
    datum_spec_with_tr = TransformFFT(
        mask_along_axis, params_masking_nonrandom_test_2
    )(datum_spec_with_tr)

    img = datum_spec.X[0, :, :, 0].numpy()
    img_with_zeros = datum_with_tr.X[0, :, :, 0].numpy()
    # first, asserting masked transform contains at least
    # one row with zeros.
    line_has_zeros = np.all((img_with_zeros == 0.0), axis=0)
    column_has_zeros = np.all((img_with_zeros == 0.0), axis=1)

    lines_with_zeros = [i for i in range(
        len(line_has_zeros)) if line_has_zeros[i]]
    columns_with_zeros = [
        i for i in range(len(column_has_zeros)) if column_has_zeros[i]
    ]
    assert(lines_with_zeros == list(range(5, 10)))
    assert(columns_with_zeros == list(range(4, 25)))

    # Second, asserting the equality of img
    # and img_with_zeros on other elements
    where_equal = [(round(img[i, j], 5) == round(img_with_zeros[i, j], 5))
                   for i in range(img.shape[0])
                   for j in range(img.shape[1])
                   if ((j not in lines_with_zeros)
                       and (i not in columns_with_zeros))]

    assert(all(where_equal))


def test_data_recovery():
    train_sample, _, _ = get_dummy_sample()
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = TransformSignal(identity)(
        TransformFFT(identity)(datum))

    assert_almost_equal(datum.X.numpy(), X, decimal=3)


def test_label_index_dict_creation():
    subset_aug_sample = [0, 1, 3, 4, 5]
    subset_aug_label = [0, 0, 0, 1, 1]
    label_index_dict = create_label_index_dict(
        subset_aug_sample, subset_aug_label)
    assert label_index_dict == \
        {0: [0, 1, 3], 1: [4, 5]}


def test_transform_construction():

    transforms = {"transform_list": [["identity"],
                                     ["mask_along_time",
                                      "identity"]]}

    constructed_transforms = [[TransformSignal(identity, transforms_args)],
                              [TransformFFT(mask_along_time,
                                            transforms_args),
                               TransformSignal(identity, transforms_args)]]

    compare(construct_transforms(transforms, transforms_args),
            constructed_transforms)


def test_delay_signal():

    train_sample, _, _ = get_dummy_sample()
    transf = TransformSignal(delay_signal, transforms_args)
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    value_cutoff = int(np.round(transforms_args["magnitude"] * X.shape[1]))
    assert_array_equal(X[:, -value_cutoff:], datum.X[:, :value_cutoff])
    assert_array_equal(X[:, :-value_cutoff], datum.X[:, value_cutoff:])


def test_em_decomposition():
    train_sample, _, _ = get_dummy_sample()
    subset_aug_sample, subset_aug_labels = get_sample(train_sample,
                                                      [["merge_two_emd"]],
                                                      sample_size=1,
                                                      random_state=1)
    transforms_args["train_sample"] = train_sample
    transforms_args["label_index_dict"] = create_label_index_dict(
        subset_aug_sample, subset_aug_labels)
    transf = TransformSignal(
        merge_two_emd,
        transforms_args)
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    assert_array_equal(X, datum.X)


def test_signal_addition():
    train_sample, _, _ = get_dummy_sample()
    subset_aug_sample, subset_aug_labels = get_sample(train_sample,
                                                      [["merge_two_signals"]],
                                                      sample_size=1,
                                                      random_state=1)
    transforms_args["train_sample"] = train_sample
    transforms_args["label_index_dict"] = create_label_index_dict(
        subset_aug_sample, subset_aug_labels)
    transf = TransformSignal(
        merge_two_signals,
        transforms_args)
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    assert_array_almost_equal(X, datum.X, 5)


def test_noise_addition():
    transforms_args["magnitude"] = 0.1
    train_sample, _, _ = get_dummy_sample()
    transf = TransformSignal(add_noise_to_signal, transforms_args)
    X = train_sample[0][0]
    scale = torch.mean(torch.abs(X)).item()
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    remains = (
        datum.X - X
    )/(transforms_args["magnitude"]*0.1)
    assert_almost_equal(np.mean(remains.numpy())/scale, 0, 1)
    assert_almost_equal(np.var(remains.numpy())/(scale*scale), 1, 1)


def test_randaugment():

    train_sample, _, _ = get_dummy_sample()
    subset_aug_sample, subset_aug_labels = get_sample(train_sample,
                                                      [["randaugment"]],
                                                      sample_size=1,
                                                      random_state=1)
    transforms_args["train_sample"] = train_sample
    transforms_args["label_index_dict"] = create_label_index_dict(
        subset_aug_sample, subset_aug_labels)
    transf = TransformSignal(randaugment, transforms_args)
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)


def test_add_noise_to_signal_with_proba():
    transforms_args["magnitude"] = 0.1
    train_sample, _, _ = get_dummy_sample()
    transf = TransformSignal(add_noise_to_signal_with_proba, transforms_args)
    random.seed(1)  # returns 0.13
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    assert_almost_equal(datum.X.numpy(), X)
    random.seed(0)  # return 0.88
    datum = transf(datum)
    scale = torch.mean(torch.abs(X)).item()
    remains = (
        datum.X - X
    )/(transforms_args["magnitude"]*0.1)
    assert_almost_equal(np.mean(remains.numpy())/scale, 0, 1)
    assert_almost_equal(np.var(remains.numpy())/(scale*scale), 1, 1)


def test_add_noise_to_signal_only_one_signal():
    transforms_args["magnitude"] = 0.1
    train_sample, _, _ = get_dummy_sample()
    transf = TransformSignal(
        add_noise_to_signal_only_one_signal, transforms_args)

    random.seed(1)  # returns 0.1
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    assert_almost_equal(datum.X[1, :].numpy(), X[1, :])

    random.seed(0)  # returns 0.8
    X = train_sample[0][0]
    datum = Datum(X=X, y=train_sample[0][1])
    datum = transf(datum)
    assert_almost_equal(datum.X[0, :].numpy(), X[0, :])
