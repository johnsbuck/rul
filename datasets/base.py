import csv
import numpy as np
from os.path import dirname
from os.path import join


def load_data(module_path, data_folder_name, data_file_name):
    """Loads data from module_path/data_folder/data_file_name.

    Args:
        module_path (str): Used to define the module path typically described with dirname(__file__).
        data_folder_name (str): Used to define directory of filename for obtaining data.
        data_file_name (str): Used to define which file to select in a given module path.
    Returns:
         Numpy Float Array
    """
    # Obtain data from file
    with open(join(module_path, 'data', data_folder_name, data_file_name)) as csv_file:
        reader = csv.reader(csv_file, delimiter=' ')
        data = list(reader)

    # Remove any blank elements in each list
    for i in xrange(len(data)):
        data[i] = filter(lambda a: a != '', data[i])

    # Return Float Array
    return np.array(data, dtype=float)


def load_phm(test_flag=True, final_flag=False):
    """Loads the PHM dataset.

    Args:
        test_flag (bool): Defines whether to include testing dataset.
            Must be scored on https://ti.arc.nasa.gov/tech/dash/pcoe/prognostic-data-repository/#phm08_challenge.
        final_flag (bool): Defines whether to include the final testing dataset used in the PHM '08 Competition.
            No key is available and no scoring site is known.

    Returns:
        Dict. Each key contains a Numpy Float Array.
    """
    module_path = dirname(__file__)
    folder = "phm"
    dataset = {}
    dataset["data"] = load_data(module_path, folder, "train.csv")
    dataset["target"] = load_data(module_path, folder, "train_rul.csv")

    if test_flag:
        dataset["test"] = load_data(module_path, folder, "test.csv")
    if final_flag:
        dataset["final"] = load_data(module_path, folder, "final_test.csv")

    return dataset


def load_cmapss(data_num=1, test_flag=True, answer_flag=False):
    """Loads a C-MAPSS dataset.

    Args:
        data_num (int): Defines what dataset should be used. Must be from 1 to 4.
        test_flag (bool): Defines whether to include testing dataset.
        answer_flag (bool): Defines whether to include answers to testing dataset.
    Returns:
        Dict. Each key contains a Numpy Float Array.
    """
    if type(data_num) != int or (data_num > 4 or data_num < 1):
        raise ValueError("The parameter, data_num, must be an integer >= 1 and <= 4.")

    module_path = dirname(__file__)
    folder = "cmapss_" + str(data_num)
    dataset = {}
    dataset["data"] = load_data(module_path, folder, "train.csv")
    dataset["target"] = load_data(module_path, folder, "train_rul.csv")

    if test_flag:
        dataset["test"] = load_data(module_path, folder, "test.csv")
    if answer_flag:
        dataset["test_target"] = load_data(module_path, folder, "test_rul.csv")

    return dataset


def load_all(test_flag=True, phm_final_flag=False, cmapss_answer_flag=False):
    """Loads all C-MAPSS and PHM datasets.

    Args:
        test_flag (bool): Defines whether to include testing dataset.
        phm_final_flag (bool): Defines whether to include the final testing dataset used in the PHM '08 Competition. No key is available and no scoring site is known.
        cmapss_answer_flag (bool): Defines whether to include answers to testing dataset.

    Returns:
        Dict of Dicts.
        Each dict has a key defined by their dataset name (i.e. 'phm'), with each dataset
        containing keys with Numpy Float Arrays
    """
    datasets = {}
    datasets["phm"] = load_phm(test_flag, phm_final_flag)
    for i in xrange(1, 5):
        datasets["cmapss_" + str(i)] = load_cmapss(i, test_flag, cmapss_answer_flag)
    return datasets
