from epoch_constructor import EpochGenerator
from braindecode.datasets.mne import (create_from_mne_epochs)


class DataImporter:

    def __init__(self, config):
        self.epochs_list = []
        self.dataset = []
        self.config = config

    def _import_epochs_list(self):
        for indiv in self.config["infos"]["list_indiv"]:
            raw_path = (self.config["infos"]["data_path"]
                        + indiv
                        + "/passive/passive_raw.fif")
            empty_raw_path = (self.config["infos"]["data_path"]
                              + indiv + "/emptyroom_"
                              + indiv + ".fif")
            self.config["infos"]["raw_path"] = raw_path
            self.config["infos"]["empty_raw_path"] = empty_raw_path
            new_ep_gen = EpochGenerator(self, self.config)
            new_ep_gen.compute_epochs()
            self.epochs_list.append(new_ep_gen.return_epochs())

    def _create_braindecode_dataset(self):
        self.windows_datasets = \
            create_from_mne_epochs(self.epochs_list,
                                   window_size_samples=50,
                                   window_stride_samples=50,
                                   drop_last_window=False)

    def compute(self):
        self._import_epoch_list()
        self._create_braindecode_dataset()

    def get(self):
        return(self.windows_datasets)
