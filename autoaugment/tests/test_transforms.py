import matplotlib.pyplot as plt
import torch
import mne
from braindecode.datasets.transform_classes import TransformSignal, \
    TransformFFT
from autoaugment.transforms.masking import mask_along_axis_random
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.learning_curve import plot_result
from autoaugment.compute_all import main_compute
from autoaugment.retrieve_data import get_dummy_sample, get_sample
import unittest
from unittest import assertEqual
import numpy as np
from ..retrieve_data import get_dummy_sample
from ..transforms.masking import mask_along_axis


class TestTransform(unittest.TestCase):
    def test_mask_along_axis_nonrandom(self):

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

        X = torch.from_numpy(train_sample[0][0])

        spec = TransformFFT(identity).transform(X)
        spec_with_tr = TransformFFT(
            mask_along_axis, params_masking_nonrandom_test_1
        ).transform(X)
        spec_with_tr = TransformFFT(
            mask_along_axis, params_masking_nonrandom_test_2
        ).transform(spec_with_tr)

        img = spec[0, :, :, 0]
        img_with_zeros = spec_with_tr[0, :, :, 0]
        # first, asserting masked transform contains at least
        # one row with zeros.

        line_has_zeros = np.all((img_with_zeros == 0.0), axis=0)
        column_has_zeros = np.all((img_with_zeros == 0.0), axis=1)

        lines_with_zeros = [i for i in range(
            len(line_has_zeros)) if line_has_zeros[i]]
        columns_with_zeros = [
            i for i in range(len(line_has_zeros)) if column_has_zeros[i]
        ]
        assertEqual(lines_with_zeros == list(range(5, 10)))
        assertEqual(columns_with_zeros == list(range(4, 25)))

        # Second, asserting the equality of img
        # and img_with_zeros on other elements

        assert([img[i, j] == img_with_zeros[i, j]
                for i in range(img.shape[0])
                for j in range(img.shape[1])
                if ((i not in lines_with_zeros)
                    and (j not in columns_with_zeros))].all())

    def test_get_sample(self):
        train_sample, _ = get_dummy_sample()

    for i in range(len(train_sample)):
        X = torch.from_numpy(train_sample[i][0])
        params_masking = {"mask_value": 0.0, "mask_param": 10, "axis": 2}
        spec = TransformFFT(identity).transform(X)
        plt.imshow(spec[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_without_tr_"
            + str(i)
            + ".png"
        )
        spec_with_tr = TransformFFT(
            mask_along_axis_random, params_masking).transform(X)
        plt.imshow(spec_with_tr[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_with_tr_" +
            str(i) + ".png"
        )

    def test_split(self):
        s = "hello world"
        self.assertEqual(s.split(), ["hello", "world"])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == "__main__":
    unittest.main()


mne.set_log_level("WARNING")


def unit_test_mask_along_axis():
    train_sample, _ = get_dummy_sample()

    for i in range(len(train_sample)):
        X = torch.from_numpy(train_sample[i][0])
        params_masking = {"mask_value": 0.0, "mask_param": 10, "axis": 2}
        spec = TransformFFT(identity).transform(X)
        plt.imshow(spec[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_without_tr_"
            + str(i)
            + ".png"
        )
        spec_with_tr = TransformFFT(
            mask_along_axis_random, params_masking).transform(X)
        plt.imshow(spec_with_tr[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_with_tr_" +
            str(i) + ".png"
        )


def unit_test_get_sample():
    params_masking = {"mask_value": 0.0, "mask_param": 10, "axis": 2}
    train_sample, _ = get_dummy_sample()
    new_transform_list = [
        [
            TransformFFT(mask_along_axis_random, params_masking),
            TransformSignal(identity),
        ],
        [TransformSignal(identity)],
    ]
    train_sample.change_transform_list(new_transform_list)
    subset = get_sample(train_sample, 1, random_state=1)

    for i in range(len(subset)):
        X = subset[i][0]
        spec = TransformFFT(identity).transform(X)
        plt.imshow(spec[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_subset_transf_"
            + str(i)
            + ".png"
        )
    import ipdb

    ipdb.set_trace()
    print("finished !")


if __name__ == "__main__":
    unit_test_get_sample()
