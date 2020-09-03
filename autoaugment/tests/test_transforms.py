import numpy as np
from numpy.testing import assert_almost_equal

from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.transforms.identity import identity
from autoaugment.retrieve_data import get_dummy_sample
from autoaugment.transforms.masking import mask_along_axis


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

    train_sample, _ = get_dummy_sample()

    X = train_sample[0][0]

    spec = TransformFFT(identity).transform(X)
    spec_with_tr = TransformFFT(
        mask_along_axis, params_masking_nonrandom_test_1
    ).transform(X)
    spec_with_tr = TransformFFT(
        mask_along_axis, params_masking_nonrandom_test_2
    ).transform(spec_with_tr)

    img = spec[0, :, :, 0].numpy()
    img_with_zeros = spec_with_tr[0, :, :, 0].numpy()
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
    train_sample, _ = get_dummy_sample()
    X = train_sample[0][0]
    X_bar = TransformSignal(identity).transform(
        TransformFFT(identity).transform(X)).numpy()
    assert_almost_equal(X_bar, X, decimal=3)
