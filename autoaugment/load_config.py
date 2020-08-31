import yaml
from braindecode.datasets.transform_classes \
    import TransformSignal, TransformFFT
from .transforms import identity, identity_ml, mask_along_axis, \
    mask_along_axis_random


dispatcher = {'identity': identity,
              'identity_ml': identity_ml,
              'mask_along_axis': mask_along_axis,
              'mask_along_axis_random': mask_along_axis_random,
              'TransformSignal': TransformSignal,
              'TransformFFT': TransformFFT}


def import_from_config(config_filename):

    with open(config_filename) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return(config)


def reconstruct_transform_list():

    
