from setuptools import setup, find_packages
setup(name='autoaugment',
      version='0.1.dev',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'torch',
          'braindecode @ git+https://github.com/Simon-Free/braindecode@experimentation_data_aug'
      ],
      )