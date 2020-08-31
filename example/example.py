import mne
import yaml
from autoaugment.retrieve_data import get_epochs_data
from autoaugment.main import main_compute
from autoaugment.learning_curve import plot_result
from autoaugment.utils import import_from_config

mne.set_log_level("WARNING")

if __name__ == "__main__":
    config = import_from_config('configs_example/config_example_raw.yaml')

    train_sample, test_sample = get_epochs_data()

    main_compute(config["models_args_list"], config["datasets_args_list"],
                 train_sample, test_sample, config["sample_size_list"],
                 config["saving_params"])
    plot_result(config["saving_params"])
