import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler
from typing import List
from datetime import datetime
from ..utils.utils import update_config_file


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new features to the DataFrame based on existing columns.

    This function adds two new features:
    - OWN_CASA: Indicates whether the customer owns any CASA based on
    the values of MTHCASA, MAXCASA, and MINCASA columns. If all three columns are zero, OWN_CASA is set to 0,
    otherwise set to 1.
    - OWN_TD: Indicates whether the customer owns any TD based on the values of MTHTD and MAXTD
    columns. If both columns are zero, OWN_TD is set to 0, otherwise set to 1.
    - OWN_CC: Indicates whether the customer owns any credit cards, based on CC_LMT_copy column,
    which is copy of original credit card limit column. If column is NaN, implies no credit card OWN_CC is set to 0,
    otherwise set to 1.

    Args:
        df (pd.DataFrame): Input DataFrame to which new features will be added.

    Returns:
        pd.DataFrame: DataFrame with new features added.
    """
    df["OWN_CASA"] = np.where(
        (df["MTHCASA"] == 0) & (df["MAXCASA"] == 0) & (df["MINCASA"] == 0), 0, 1
    )

    df["OWN_TD"] = np.where((df["MTHTD"] == 0) & (df["MAXTD"] == 0), 0, 1)

    df["OWN_CC"] = np.where(df["CC_LMT_copy"].isna(), 0, 1)

    df["OWN_PREV_CC"] = np.where(df["CC_AVE_copy"].isna(), 0, 1)

    # Drops *_copy which was used to generate binary features
    df = df.drop(
        columns=[
            "CC_LMT_copy",
            "CC_AVE_copy",
        ]
    )

    return df


def standardize_columns(
    df: pd.DataFrame,
    num_columns: List[str],
    k: int = 10,
    scaler_folder: str = "scalers",
) -> pd.DataFrame:
    """
    Perform standardization on specified numerical columns. The function saves the scalers to allow
    reapplication on new data, ensuring consistent scaling.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_columns (List[str]): List of numerical column names to be standardized.
        k (int): This parameter is included for consistency with the function signature but is not used.
        scaler_folder (str): Folder to save scaler files. Default is 'scalers'.

    Returns:
        pd.DataFrame: DataFrame with the original numerical columns replaced by the standardized columns.
    """
    scalers = {}

    new_df = df.copy()

    for col in num_columns:
        scaler = StandardScaler()

        scaled_col_name = f"{col}_scaled"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        scaler_file = os.path.join(scaler_folder, f"{col}_scaler_{timestamp}.pkl")

        new_df[scaled_col_name] = scaler.fit_transform(df[[col]].to_numpy())

        joblib.dump(scaler, scaler_file)

        scalers[col] = scaler_file

    new_df = new_df.drop(columns=num_columns)

    scaler_config = {"scalers": scalers}

    update_config_file("maybank/conf/base/scaler.yaml", scaler_config)

    return new_df


def stratified_kfold_target_encoding(
    df: pd.DataFrame,
    cat_columns: List[str],
    target_column: str,
    k: int = 10,
    encoder_folder: str = "encoders",
) -> pd.DataFrame:
    """
    Perform k-fold target encoding using Stratified K-Fold on the specified categorical columns.
    Stratified K-Fold is to prevent data leakage and ensure each fold is representative of the whole dataset.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cat_columns (List[str]): List of categorical variable column names.
        target_column (str): Name of the target variable column.
        k (int): Number of folds for cross-validation. Default is 10.
        encoder_folder (str): Folder to save encoder files. Default is 'encoders'.

    Returns:
        pd.DataFrame: DataFrame with the original categorical columns replaced by the new encoded columns.
    """
    encoders = {}

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=98)

    new_df = df.copy()

    for col in cat_columns:
        encoder = TargetEncoder()

        encoded_col_name = f"{col}_encoded"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        encoder_file = os.path.join(encoder_folder, f"{col}_encoder_{timestamp}.pkl")

        new_df[encoded_col_name] = np.nan

        for fold, (train_index, val_index) in enumerate(
            kf.split(df, df[target_column])
        ):
            train_fold = df.iloc[train_index]
            val_fold = df.iloc[val_index]

            encoder.fit_transform(train_fold[[col]], train_fold[target_column])

            new_df.loc[val_index, encoded_col_name] = encoder.transform(
                val_fold[[col]]
            ).ravel()

        joblib.dump(encoder, encoder_file)

        encoders[col] = encoder_file

    new_df = new_df.drop(columns=cat_columns)

    encoder_config = {"encoders": encoders}

    update_config_file("maybank/conf/base/encoder.yaml", encoder_config)

    return new_df
