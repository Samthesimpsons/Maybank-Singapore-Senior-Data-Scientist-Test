from typing import List, Tuple
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd


def resample_data(
    df: pd.DataFrame, target_column: str, categorical_features: List[str]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply RandomUnderSampler and SMOTENC to the DataFrame to balance the dataset.
    Will apply undersampling first before oversampling.

    Args:
        df (pd.DataFrame): DataFrame containing features and the target variable.
        target_column (str): The name of the target column.
        categorical_features (List[str]): List of column names for categorical features.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the resampled features DataFrame and
            the resampled target Series.
    """
    categorical_indices = [df.columns.get_loc(col) for col in categorical_features]
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    smote_nc = SMOTENC(
        categorical_features=categorical_indices,
        random_state=98,
        sampling_strategy="minority",
    )
    X_res, y_res = smote_nc.fit_resample(X, y)

    df_resampled = pd.concat(
        [
            pd.DataFrame(X_res, columns=X.columns),
            pd.DataFrame(y_res, columns=[target_column]),
        ],
        axis=1,
    )

    X_resampled = df_resampled.drop(target_column, axis=1)
    y_resampled = df_resampled[target_column]

    rus = RandomUnderSampler(random_state=98, sampling_strategy="majority")
    X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

    return X_resampled, y_resampled
