import sweetviz as sv
import pandas as pd
from typing import List, Optional
from scipy import stats


def convert_c_seg_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the 'C_seg' column in the DataFrame to binary.

    Args:
        df (pd.DataFrame): Input DataFrame containing the 'C_seg' column.

    Returns:
        pd.DataFrame: DataFrame with the 'C_seg' column converted to binary:
                      - 1 if 'C_seg' is 'AFFLUENT'.
                      - 0 otherwise.
    """
    df_copy = df.copy()

    df_copy["C_seg"] = df_copy["C_seg"].apply(lambda x: 1 if x == "AFFLUENT" else 0)

    return df_copy


def perform_eda_with_sweetviz(
    df: pd.DataFrame,
    target_feat: Optional[str] = None,
    html_file_path: Optional[str] = None,
) -> None:
    """
    Perform Exploratory Data Analysis (EDA) using Sweetviz library and optionally save the report to an HTML file.

    Args:
        df (pandas.DataFrame): The DataFrame containing the dataset for analysis.
        target_feat (str, optional): The name of the target feature. Defaults to None.
        html_file_path (str, optional): Path to save the HTML report. Defaults to None.
    """

    eda_report = sv.analyze(df, target_feat=target_feat)
    if html_file_path:
        eda_report.show_html(filepath=html_file_path)
    else:
        eda_report.show_notebook()


def drop_outliers(
    df: pd.DataFrame,
    num_columns: List[str],
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Drop outliers from specified numerical columns using z-score.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_columns (List[str]): List of numerical column names to be processed.
        threshold (float): Z-score threshold for outlier detection. Default is 3.0.

    Returns:
        pd.DataFrame: DataFrame with outliers removed from specified columns.
    """

    new_df = df.copy()

    curr_len = len(new_df)
    curr_len_1_label = len(new_df[new_df["C_seg"] == 1])
    curr_len_0_label = len(new_df[new_df["C_seg"] == 0])

    for col in num_columns:
        z_scores = stats.zscore(new_df[col])

        old_len = len(new_df)

        abs_z_scores = abs(z_scores)

        filtered_entries = abs_z_scores < threshold

        new_df = new_df[filtered_entries]

        print(f"Dropped {old_len-len(new_df)} observations for feature: {col}")

    new_len = len(new_df)
    new_len_1_label = len(new_df[new_df["C_seg"] == 1])
    new_len_0_label = len(new_df[new_df["C_seg"] == 0])

    print(
        f"Total: Dropped {curr_len-new_len} observations, {curr_len_1_label-new_len_1_label} for 1, {curr_len_0_label-new_len_0_label} for 0 label."
    )

    print(
        f"New 1 lable count: {new_len_1_label}, New 0 label count: {new_len_0_label}."
    )

    return new_df


def impute_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing missing values.

    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    df["C_EDU"] = df["C_EDU"].fillna("Not Provided")
    df["C_HSE"] = df["C_HSE"].fillna("Not Provided")
    df["gn_occ"] = df["gn_occ"].fillna("Not Provided")
    df["CC_AVE_copy"] = df[
        "CC_AVE"
    ].copy()  # To be used in feature engineering step later
    df["CC_LMT_copy"] = df[
        "CC_LMT"
    ].copy()  # To be used in feature engineering step later

    incm_typ_mode = df["INCM_TYP"].mode()[0]
    df["INCM_TYP"] = df["INCM_TYP"].fillna(incm_typ_mode)

    zero_impute_cols = [
        "PC",
        "CASATD_CNT",
        "MTHCASA",
        "MAXCASA",
        "MINCASA",
        "DRvCR",
        "MTHTD",
        "MAXTD",
        "UT_AVE",
        "MAXUT",
        "N_FUNDS",
        "pur_price_avg",
        "MAX_MTH_TRN_AMT",
        "MIN_MTH_TRN_AMT",
        "AVG_TRN_AMT",
        "ANN_TRN_AMT",
        "ANN_N_TRX",
        "HL_tag",
        "AL_tag",
        "CC_LMT",
        "CC_AVE",
    ]
    for col in zero_impute_cols:
        df[col] = df[col].fillna(0)

    return df


def convert_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert float64 columns to float32 for efficiency.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with float64 columns converted to float32.
    """
    float64_cols = df.select_dtypes(include=["float64"]).columns
    for col in float64_cols:
        df[col] = df[col].astype("float32")
    return df
