import mne


class EpochGenerator:

    def __init__(self, config):
        self.config = config
        self.raw = None
        self.empty_raw = None
        self.events = None
        self.final_epochs = None
        self.ica = None
        self.reject = None

    def _import_data(self):
        self.raw = mne.io.read_raw_fif(self.raw_path)
        self.empty_raw = mne.io.read_raw_fif(self.empty_raw_path)

    def _filter_data(self):
        self.raw.fix_mag_coil_types()
        self.raw = \
            mne.preprocessing.maxwell_filter(self.raw,
                                             calibration=self.config["info"]["cal_file_path"])
        self.raw.filter(l_freq=0.5, h_freq=40)
        self.empty_raw.fix_mag_coil_types()
        self.empty_raw = \
            mne.preprocessing.maxwell_filter(self.raw,
                                             calibration=self.config["info"]["cal_file_path"],
                                             coord_frame="meg")
        self.empty_raw.filter(l_freq=0.5, h_freq=40)

    def _build_epochs_evoked(self):
        self.events = mne.find_events(self.raw, stim_channel='STI101')
        self.raw.info['projs'] = list()  # remove proj, don't proj while interpolating
        self.epochs = mne.Epochs(self.raw,
                                 self.events,
                                 self.config["event_id"],
                                 self.config["tmin"],
                                 self.config["tmax"],
                                 baseline=(None, 0),
                                 reject=None,
                                 verbose=False,
                                 detrend=0,
                                 preload=True)
        self.reject = get_rejection_threshold(epochs)

    def _create_epochs(raw):
        self.ica = mne.preprocessing.ICA(n_components=30)
        self.ica.fit(raw)
        # raw.load_data()
        self.ica.exclude = []
        eog_indices, eog_scores = self.ica.find_bads_eog(self.raw)
        ecg_indices, ecg_scores = self.ica.find_bads_ecg(self.raw, method='correlation')
        self.ica.exclude = ecg_indices + eog_indices
        self.epochs = mne.Epochs(self.raw, events, event_id, tmin, tmax,
                                 baseline=(None, 0), reject=reject,
                                 verbose=False, detrend=0, preload=True)
    
    def compute(self):
        self._import_data()
        self._filter_data()
        self._build_epochs_evoked()
        self._create_epochs()
        
    def get(self):
        return(self.epochs)