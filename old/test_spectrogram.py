import unittest
import inspect
import mne
import transform_side_funcs
import matplotlib.pyplot as plt
from ml_vs_dl import get_epochs_data
import torch
import torchaudio
from braindecode.datasets import create_from_mne_epochs


def mask_along_axis(
        specgram,
        mask_start: int,
        mask_end: int,
        mask_value: float,
        axis: int
):
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``, and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_param (int): Number of columns to be masked will be uniformly sampled from [0, mask_param]
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)

    Returns:
        Tensor: Masked spectrogram of dimensions (channel, freq, time)
    """

    mask_start = (torch.tensor(mask_start).long()).squeeze()
    mask_end = (torch.tensor(mask_end).long()).squeeze()

    if axis == 1:
        specgram[:, mask_start:mask_end, :, :] = mask_value
    elif axis == 2:
        specgram[:, :, mask_start:mask_end, :] = mask_value
    else:
        raise ValueError('Only Frequency and Time masking are supported')
    print(specgram)
    specgram = specgram.reshape(specgram.shape[:-2] + specgram.shape[-2:])
    return specgram


class Policy:

    def __init__(self, policy_type: str, args):
        self.policy_type = policy_type
        self.args = args

    def apply(self, spectrogram):
        if self.policy_type == "axis_mask":
            return(mask_along_axis(spectrogram, self.args["mask_start"], self.args["mask_end"], self.args["mask_value"], self.args["axis"]))
        # if type = "axis_warp":
        #    return(warp_along_axis())


def tensor_to_img(spectrogram):  # arbitrary, looks good on my screen.
    for i in range(spectrogram.shape[0]):
        plt.imshow(spectrogram[i], cmap='YlGn', interpolation='sinc', vmin=-100, vmax=100)
        plt.show()
    print(spectrogram.shape)


def transform(window, policy_list):
    n_fft = 512
    hop_length = 64
    win_length = n_fft
    spectrogram = torch.stft(window, n_fft=n_fft, 
                             hop_length=hop_length,
                             win_length=win_length,
                             window=torch.hann_window(n_fft))

    for policy in policy_list:
        spectrogram = policy.apply(spectrogram)

    to_plot = torch.norm(spectrogram, dim=3)
    to_plot = torchaudio.transforms.AmplitudeToDB().forward(to_plot)

    tensor_to_img(to_plot)
    augmented_window = torchaudio.functional.istft(
        spectrogram,    
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(n_fft))
    return(spectrogram)


class TestSpectrogram(unittest.TestCase):
    def test_epoch(self):
        epochs_train_list, epochs_test_list = get_epochs_data([0],[1])

#         train_sample = create_from_mne_epochs(
#             epochs_train_list,
#             window_size_samples=3000,
#             window_stride_samples=3000,
#             drop_last_window=False)

        test_sample = create_from_mne_epochs(
            epochs_test_list, window_size_samples=3000,
            window_stride_samples=3000,
            drop_last_window=False)

        test_sample.datasets[0].windows.load_data()
        test = torch.tensor(test_sample.datasets[0].windows._data[0])
        args = {"mask_start": 10,
                "mask_end": 40,
                "mask_value": 1,
                "axis": 2}
        policy_type = "axis_mask"
        policy = Policy(policy_type, args)
        spec = transform(test, [policy])


if __name__ == "__main__":
    unittest.main()
