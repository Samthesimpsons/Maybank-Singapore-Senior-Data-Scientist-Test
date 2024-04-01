import yaml
import pandas as pd
from typing import Dict, Any


def load_catalog(catalog_file_path: str) -> Dict[str, Any]:
    """
    Load catalog data from a YAML file.

    Args:
        catalog_file_path (str): Path to the catalog YAML file.

    Returns:
        dict: Loaded catalog data.
    """
    with open(catalog_file_path, "r") as file:
        catalog_data = yaml.safe_load(file)
    return catalog_data


def load_raw_data(catalog_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw data using information from the catalog.

    Args:
        catalog_data (dict): Catalog data containing information about the raw data.

    Returns:
        pandas.DataFrame: Loaded raw data.
    """
    raw_data_path = catalog_data["raw_data"]["filepath"]
    raw_data = pd.read_csv(raw_data_path)
    return raw_data


def load_meta_data(catalog_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Load meta data using information from the catalog.

    Args:
        catalog_data (dict): Catalog data containing information about the meta data.

    Returns:
        pandas.DataFrame: Loaded meta data.
    """
    meta_data_path = catalog_data["meta_data"]["filepath"]
    meta_data = pd.read_csv(meta_data_path)
    return meta_data


def load_processed_data(catalog_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Load processed data using information from the catalog.

    Args:
        catalog_data (dict): Catalog data containing information about the processsed data.

    Returns:
        pandas.DataFrame: Loaded processed data.
    """
    processed_data_path = catalog_data["processed_data"]["filepath"]
    processed_data = pd.read_csv(processed_data_path, index_col=None)
    return processed_data
