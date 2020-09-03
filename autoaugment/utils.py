import os
import getpass
from pathlib import Path


def update_saving_params(saving_params):
    saving_params["main_save_folder"] = get_user_folder(saving_params)

    saving_params["result_dict_save_folder"] = os.path.join(
        saving_params["main_save_folder"], "result_dict")
    Path(saving_params["result_dict_save_folder"]).mkdir(parents=True,
                                                         exist_ok=True)

    saving_params["plots_save_folder"] = os.path.join(
        saving_params["main_save_folder"],
        "plots")
    Path(saving_params["plots_save_folder"]).mkdir(parents=True, exist_ok=True)
    return(saving_params)


def get_user_folder(saving_params):

    users_with_custom_dir = saving_params["folder"].keys()

    # smart user detection, avoid having to enter username each time
    # (pretty fastidious)
    for user in users_with_custom_dir:
        if user in os.getcwd():
            return saving_params["folder"][user]

    # if not found:
    user = getpass.getuser()
    if user in saving_params["folder"].keys():
        return saving_params["folder"][user]

    # if still no results
    default_dir = os.path.join(os.getcwd(), "result_folder")
    Path(default_dir).mkdir(parents=True, exist_ok=True)

    return default_dir
