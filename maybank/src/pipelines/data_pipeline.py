from ..data_processing.data_loading import load_catalog, load_raw_data
from ..data_processing.data_preprocessing import (
    convert_c_seg_to_binary,
    impute_missing_data,
    convert_float64_to_float32,
)
from ..feature_engineering.feature_engineering import (
    add_features,
    stratified_kfold_target_encoding,
)
from ..utils.utils import save_processed_data_as_csv
import time


def run_data_pipeline() -> str:
    """Runs the data pipeline and returns a message indicating its completion.

    Returns:
        str: A message indicating the completion of the data pipeline.
    """
    start_time = time.time()

    # Load catalog configuration and raw data
    load_catalog_start = time.time()
    catalog_config = load_catalog("maybank/conf/base/catalog.yaml")
    raw_data = load_raw_data(catalog_config)
    load_catalog_duration = time.time() - load_catalog_start
    print(f"Load Catalog Duration: {round(load_catalog_duration, 2)} seconds")

    # Preprocess data
    preprocess_start = time.time()
    raw_data = convert_c_seg_to_binary(raw_data)
    cleaned_data = impute_missing_data(raw_data)
    cleaned_data = convert_float64_to_float32(cleaned_data)
    cleaned_data = cleaned_data.drop(columns=["C_ID", "PC", "CC_AVE"]).reset_index(
        drop=True
    )
    preprocess_duration = time.time() - preprocess_start
    print(f"Preprocessing Duration: {round(preprocess_duration, 2)} seconds")

    # Feature engineering
    feature_engineering_start = time.time()
    processed_data = add_features(cleaned_data)
    processed_data = stratified_kfold_target_encoding(
        processed_data,
        ["INCM_TYP", "C_EDU", "C_HSE", "gn_occ"],
        "C_seg",
        encoder_folder="maybank/data/models/",
    )
    feature_engineering_duration = time.time() - feature_engineering_start
    print(
        f"Feature Engineering Duration: {round(feature_engineering_duration, 2)} seconds"
    )

    # Save processed data to CSV
    save_data_start = time.time()
    save_processed_data_as_csv(
        processed_data, "maybank/data/preprocessed/processed_data.csv"
    )
    save_data_duration = time.time() - save_data_start
    print(f"Saving Data Duration: {round(save_data_duration, 2)} seconds")

    total_duration = time.time() - start_time
    print(f"Total Duration: {round(total_duration, 2)} seconds")

    return "Data Pipeline finished."
