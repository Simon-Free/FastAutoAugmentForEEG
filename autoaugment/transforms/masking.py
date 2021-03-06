import torch


def mask_along_axis(datum, params):
    r"""
    Apply a mask along ``axis``. Mask will be applied from indices
    ``[v_0, v_0 + v)``, where
    ``v`` is sampled from ``uniform(0, mask_param)``,
    and ``v_0`` from ``uniform(0, max_v - v)``.
    All examples will have the same mask interval.

    Args:
        specgram (Tensor): Real spectrogram (channel, freq, time)
        mask_start (int): First column masked
        mask_end (int): First column unmasked
        mask_value (float): Value to assign to the masked columns
        axis (int): Axis to apply masking on (1 -> frequency, 2 -> time)

    Returns:
        Tensor: Masked spectrogram of dimensions (channel, freq, time)
    """
    specgram = datum.X

    mask_start = params["mask_start"]
    mask_end = params["mask_end"]

    if params["axis"] == 1:
        specgram[:, mask_start:mask_end, :, :] = params["mask_value"]
    elif params["axis"] == 2:
        specgram[:, :, mask_start:mask_end, :] = params["mask_value"]
    else:
        raise ValueError('Only Frequency and Time masking are supported')
    specgram = specgram.reshape(specgram.shape[:-2] + specgram.shape[-2:])
    datum.X = specgram
    return datum


def mask_along_axis_random(datum, params):

    specgram = datum.X
    value = torch.rand(1) \
        * params['magnitude'] * specgram.size(params["axis"])

    min_value = torch.rand(1) * (specgram.size(params["axis"]) - value)

    params["mask_start"] = (min_value.long()).squeeze()
    params["mask_end"] = (min_value.long() + value.long()).squeeze()

    datum = mask_along_axis(datum, params)
    return datum


def mask_along_time(datum, params):

    params_time = params["masking_along_time"]
    params_time["magnitude"] = params["magnitude"]

    datum = mask_along_axis_random(datum, params_time)
    return datum


def mask_along_frequency(datum, params):

    params_frequency = params["masking_along_frequency"]
    params_frequency["magnitude"] = params["magnitude"]

    datum = mask_along_axis_random(datum, params_frequency)
    return datum
