from autoaugment.retrieve_data import get_dummy_sample, get_sample
from autoaugment.compute_all import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.transforms.identity import identity, identity_ml
from autoaugment.transforms.masking import mask_along_axis_random
from braindecode.datasets.transform_classes import TransformSignal, TransformFFT
import mne
import torch
import matplotlib.pyplot as plt
mne.set_log_level("WARNING")


def unit_test_mask_along_axis():
    train_sample, _ = get_dummy_sample()

    for i in range(len(train_sample)):
        X = torch.from_numpy(train_sample[i][0])
        params_masking = {"mask_value": 0.0,
                          "mask_param": 10,
                          "axis": 2}
        spec = TransformFFT(identity).transform(X)
        plt.imshow(spec[0, :, :, 0])
        plt.savefig("/storage/store/work/sfreybur/img/spec_test/spec_without_tr_"
                    + str(i) + ".png")
        spec_with_tr = TransformFFT(mask_along_axis_random, params_masking).transform(X)
        plt.imshow(spec_with_tr[0, :, :, 0])
        plt.savefig("/storage/store/work/sfreybur/img/spec_test/spec_with_tr_"
                    + str(i) + ".png")


def unit_test_get_sample():
    params_masking = {"mask_value": 0.0,
                      "mask_param": 10,
                      "axis": 2}
    train_sample, _ = get_dummy_sample()
    new_transform_list = [
        [TransformFFT(mask_along_axis_random, params_masking),
         TransformSignal(identity)], [TransformSignal(identity)]]
    train_sample.change_transform_list(new_transform_list)
    subset = get_sample(train_sample, 1, random_state=1)

    for i in range(len(subset)):
        X = subset[i][0]
        spec = TransformFFT(identity).transform(X)
        plt.imshow(spec[0, :, :, 0])
        plt.savefig(
            "/storage/store/work/sfreybur/img/spec_test/spec_subset_transf_"
            + str(i) + ".png")
    import ipdb; ipdb.set_trace()
    print("finished !")


if __name__ == "__main__":
    unit_test_get_sample()


