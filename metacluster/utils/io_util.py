#!/usr/bin/env python
# Created by "Thieu" at 16:10, 31/07/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import csv
from pathlib import Path


def write_dict_to_csv(data:dict, save_path=None, file_name=None):
    """
    Write a list of dictionaries to a CSV file.

    Args:
        data (list): A list of dictionaries.
        save_path (str): Path to save the file
        file_name (str): The name of the output CSV file.

    Returns:
        None
    """
    save_file = f"{save_path}/{file_name}.csv"
    file_exists = Path(save_file)

    with open(save_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    return None
