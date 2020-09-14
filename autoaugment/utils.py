import os
from pathlib import Path


def update_saving_params(saving_params):
    Path(saving_params["main_save_folder"]).mkdir(parents=True,
                                                  exist_ok=True)

    saving_params["result_dict_save_folder"] = os.path.join(
        saving_params["main_save_folder"], "result_dict")
    Path(saving_params["result_dict_save_folder"]).mkdir(parents=True,
                                                         exist_ok=True)

    saving_params["plots_save_folder"] = os.path.join(
        saving_params["main_save_folder"],
        "plots")
    Path(saving_params["plots_save_folder"]).mkdir(parents=True, exist_ok=True)
    return(saving_params)
