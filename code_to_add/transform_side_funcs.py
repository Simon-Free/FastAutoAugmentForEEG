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
        
args= {"mask_start": 10,
       "mask_end": 40,
       "mask_value": 1,
       "axis": 2}
policy_type = "axis_mask"
policy = Policy(policy_type, args)
spec = transform(test, [policy])