from setuptools import setup

setup(
    name='autoaugment',
    version='0.0.2',
    description='Module to autoaugment EEG data',
    author='Simon FREYBURGER',
    author_email='simon.freyburger@inria.fr',
    packages=['autoaugment'],  # same as name
    install_requires=['braindecode @ git+https://github.com/Simon-Free/' \
                      'braindecode@experimentation_data_aug',
                      'mne @ git+https://github.com/mne-tools/' \
                      'mne-python@master']
)
