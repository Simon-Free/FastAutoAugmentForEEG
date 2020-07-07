import torch


def mask_along_axis(
        specgram,
        mask_start: int,
        mask_end: int,
        mask_value: float,
        axis: int):
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices
    ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``,
    and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_param (int): Number of columns to be masked will be
        uniformly sampled from [0, mask_param]
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


class TransformFFT:

    def __init__(self, policy, fft_args):
        self.policy = policy
        self.n_fft = fft_args.fft
        self.hop_length = fft_args.hop_length
        self.win_length = fft_args.win_length

    def fit(self, X):
        pass

    def transform(self, X):
        if not (len(X.shape) == 4):
            # (len(X.shape) == 4) characterizes the
            # spectrogramm of an epoch with several
            # channels.
            X = torch.stft(X, n_fft=self.n_fft,
                           hop_length=self.hop_length,
                           win_length=self.win_length,
                           window=torch.hann_window(self.n_fft))

        return self.policy(X)
        # if type = "axis_warp":
        #    return(warp_along_axis())


class TransformSignal:

    def __init__(self, policy, fft_args): #TODO, construire un fft_args vanille
        self.policy = policy
        self.n_fft = fft_args.fft
        self.hop_length = fft_args.hop_length
        self.win_length = fft_args.win_length

    def fit(self, X):
        pass

    def transform(self, X):
        if (len(X.shape) == 4):
            X = torch.istft(X,
                            n_fft=self.n_fft,
                            hop_length=self.hop_length,
                            win_length=self.n_fft,
                            window=torch.hann_window(self.n_fft))
        return self.policy(X)
    
    

