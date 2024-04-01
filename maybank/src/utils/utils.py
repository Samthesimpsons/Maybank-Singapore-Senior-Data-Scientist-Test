import yaml
import pickle
import pandas as pd
from typing import Dict, Any


def update_config_file(file_path: str, new_data: Dict[str, Any]) -> None:
    """
    Update an existing YAML configuration file with new data.

    Args:
        file_path (str): Path to the configuration file.
        new_data (Dict[str, Any]): New data to be added to the configuration.

    Returns:
        None
    """
    try:
        with open(file_path, "r") as f:
            config = yaml.safe_load(f) or {}

        config.update(new_data)

        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    except Exception as e:
        print(f"Error occurred while updating config file: {e}")


def save_dataframe_to_csv(df: pd.DataFrame, file_path: str, **kwargs: Any) -> None:
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame will be saved.
        **kwargs: Additional keyword arguments to be passed to `to_csv` function.

    Returns:
        None
    """
    try:
        df.to_csv(file_path, index=False, **kwargs)
        print(f"DataFrame saved to {file_path}")

    except Exception as e:
        print(f"Error occurred while saving DataFrame: {e}")


def save_processed_data_as_csv(df: pd.DataFrame, file_path: str, **kwargs: Any) -> None:
    """
    Save a Pandas DataFrame to a CSV file and update configuration file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame will be saved.
        **kwargs: Additional keyword arguments to be passed to `to_csv` function.

    Returns:
        None
    """
    try:
        save_dataframe_to_csv(df, file_path, **kwargs)
        new_data = {
            "processed_data": {"type": "pandas.CSVDataSet", "filepath": file_path}
        }
        update_config_file("maybank/conf/base/catalog.yaml", new_data)

    except Exception as e:
        print(f"Error occurred while saving processed data: {e}")


def save_predicted_data_as_csv(df: pd.DataFrame, file_path: str, **kwargs: Any) -> None:
    """
    Save a Pandas DataFrame to a CSV file and update configuration file.

    Args:
        df (pd.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the DataFrame will be saved.
        **kwargs: Additional keyword arguments to be passed to `to_csv` function.

    Returns:
        None
    """
    try:
        save_dataframe_to_csv(df, file_path, **kwargs)
        new_data = {
            "predicted_data": {"type": "pandas.CSVDataSet", "filepath": file_path}
        }
        update_config_file("maybank/conf/base/catalog.yaml", new_data)

    except Exception as e:
        print(f"Error occurred while saving predicted data: {e}")


def save_model(model: Any, filename: str) -> None:
    """
    Save a machine learning model to a file using pickle.

    Args:
        model (Any): The machine learning model to be saved.
        filename (str): The path or filename where the model should be saved.

    Returns:
        None: This function doesn't return anything. It saves the model to the specified file.
    """

    with open(filename, "wb") as file:
        pickle.dump(model, file)

    print(f"Model saved to {filename}")
