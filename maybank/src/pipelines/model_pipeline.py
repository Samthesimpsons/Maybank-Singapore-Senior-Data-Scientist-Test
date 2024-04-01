from ..modeling.modeling import tune_xgboost_model
from ..data_processing.data_loading import load_catalog, load_processed_data
import time


def run_model_pipeline() -> str:
    """Runs the model pipeline and returns a message indicating its completion.

    Returns:
        str: A message indicating the completion of the modeling pipeline.
    """
    tune_model_start = time.time()

    catalog_config = load_catalog("maybank/conf/base/catalog.yaml")

    processed_data = load_processed_data(catalog_config)

    print(f"Number of rows in processed_data: {len(processed_data)}")

    paramters_config = load_catalog("maybank/conf/base/parameters.yaml")

    tune_xgboost_model(
        processed_data,
        label_column="C_seg",
        binary_columns=["HL_tag", "AL_tag", "OWN_CASA", "OWN_TD"],
        n_trials=paramters_config["n_trials"],  # For optuna number of trials
        n_splits=paramters_config["n_splits"],  # For k_fold
    )
    tune_model_duration = time.time() - tune_model_start

    print(f"Tune Model Duration: {round(tune_model_duration, 2)} seconds")

    return "Modeling Pipeline finished."
