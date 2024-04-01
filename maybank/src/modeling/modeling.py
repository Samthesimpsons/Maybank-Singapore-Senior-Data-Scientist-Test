import optuna
import pandas as pd
import xgboost as xgb
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from typing import List
from ..data_processing.data_sampling import resample_data
from ..utils.utils import save_model, update_config_file


def perform_cross_validation_over_models(
    df, label_column, binary_columns, number_folds=3
):
    """
    Perform K-fold cross-validation with resampling over 2 models:
    - Random Forest
    - XGBoost Classifier

    Args:
        df (pd.DataFrame): DataFrame containing features and label column.
        label_column (str): Name of the label column in the DataFrame.
        binary_columns (List[str]): List of column names that are binary.
        number_of_folds (int): Number of folds for k fold validation.
    """

    kf = KFold(n_splits=number_folds, shuffle=True, random_state=42)

    rf_precision_total = 0
    rf_recall_total = 0
    rf_f1_total = 0
    xgb_precision_total = 0
    xgb_recall_total = 0
    xgb_f1_total = 0

    for train_index, val_index in tqdm(kf.split(df)):
        train_df, val_df = df.iloc[train_index], df.iloc[val_index]

        X_train, y_train = train_df.drop(columns=[label_column]), train_df[label_column]
        X_val, y_val = val_df.drop(columns=[label_column]), val_df[label_column]

        X_train_resampled, y_train_resampled = resample_data(
            pd.concat([X_train, y_train], axis=1),
            label_column,
            binary_columns,
        )

        # Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_resampled, y_train_resampled)
        rf_y_pred = rf_model.predict(X_val)
        rf_precision_total += precision_score(y_val, rf_y_pred)
        rf_recall_total += recall_score(y_val, rf_y_pred)
        rf_f1_total += f1_score(y_val, rf_y_pred)

        # XGBoost model
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.001,
            n_estimators=100,
            random_state=42,
        )
        xgb_model.fit(X_train_resampled, y_train_resampled)
        xgb_y_pred = xgb_model.predict(X_val)
        xgb_precision_total += precision_score(y_val, xgb_y_pred)
        xgb_recall_total += recall_score(y_val, xgb_y_pred)
        xgb_f1_total += f1_score(y_val, xgb_y_pred)

    rf_precision_avg = rf_precision_total / number_folds
    rf_recall_avg = rf_recall_total / number_folds
    rf_f1_avg = rf_f1_total / number_folds
    xgb_precision_avg = xgb_precision_total / number_folds
    xgb_recall_avg = xgb_recall_total / number_folds
    xgb_f1_avg = xgb_f1_total / number_folds

    print("Random Forest - Avg Precision:", round(rf_precision_avg, 2))
    print("Random Forest - Avg Recall:", round(rf_recall_avg, 2))
    print("Random Forest - Avg F1 Score:", round(rf_f1_avg, 2))
    print("XGBoost - Avg Precision:", round(xgb_precision_avg, 2))
    print("XGBoost - Avg Recall:", round(xgb_recall_avg, 2))
    print("XGBoost - Avg F1 Score:", round(xgb_f1_avg, 2))


def optimize_xgboost_model(
    trial: optuna.trial.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    number_of_splits: int,
    binary_columns: List[str],
) -> float:
    """
    Optimize XGBoost model parameters using Optuna.

    Args:
        trial (optuna.trial.Trial): A trial object from Optuna.
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        number_of_splits (int): Number of splits for K-Fold cross-validation.
        binary_columns (List[str]): List of column names that are binary.

    Returns:
        float: The mean recall score across all K-Fold splits.
    """
    xgboost_parameters = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "gpu_id": 0,
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
    }

    k_fold = KFold(n_splits=number_of_splits, shuffle=True, random_state=42)
    recall_scores = []
    for train_index, test_index in k_fold.split(X):
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        processed_data = pd.concat([X_train_fold, y_train_fold], axis=1)
        resampled_X, resampled_y = resample_data(
            processed_data, y_train_fold.name, binary_columns
        )

        model = xgb.XGBClassifier(**xgboost_parameters)
        model.fit(resampled_X, resampled_y)

        y_predicted = model.predict(X_test_fold)
        recall = recall_score(y_test_fold, y_predicted)
        recall_scores.append(recall)
    return np.mean(recall_scores)


def tune_xgboost_model(
    df: pd.DataFrame,
    label_column: str,
    binary_columns: List[str],
    n_trials: int = 100,
    n_splits: int = 10,
) -> None:
    """
    Train and optimize an XGBoost model using Optuna, based on a provided DataFrame, and save the model, and also updates the model config.

    Args:
        df (pd.DataFrame): The DataFrame containing the feature set and target variable.
        label_column (str): The name of the column in the DataFrame that represents the target variable.
        binary_columns (List[str]): List of column names that are binary.
        n_trials (int): Number of trials for Optuna optimization. Defaults to 100.
        n_splits (int): Number of splits for K-Fold cross-validation. Defaults to 10.

    Returns:
        None: The function trains the model, optimizes it, and saves it to a file, and also updates the model config.
    """
    X = df.drop(label_column, axis=1)
    y = df[label_column]

    study = optuna.create_study(direction="maximize")

    # Added our best result here as enqueue for future retraining
    study.enqueue_trial(
        {
            "n_estimators": 2284,
            "max_depth": 10,
            "learning_rate": 0.02887905430507317,
            "subsample": 0.7405879685315797,
            "colsample_bytree": 0.6021896910634843,
            "gamma": 0.030145127416234097,
            "min_child_weight": 4,
        }
    )

    study.optimize(
        lambda trial: optimize_xgboost_model(trial, X, y, n_splits, binary_columns),
        n_trials=n_trials,
    )

    best_hyperparameters = study.best_params
    print("Best Hyperparameters for Maximum Recall:", best_hyperparameters)

    model = xgb.XGBClassifier(**best_hyperparameters)
    model.fit(X, y)

    # feature_importances = model.feature_importances_
    # sorted_importances = np.sort(feature_importances)
    # threshold_index = int(
    #     len(sorted_importances) * 0.10
    # )  # Select only top 90% of variables
    # threshold_value = sorted_importances[threshold_index]

    # feature_selector = SelectFromModel(estimator=model, threshold=threshold_value)
    # feature_selector.fit(X, y)
    # X_selected = feature_selector.transform(X)

    # dropped_features = X.columns[~feature_selector.get_support()]
    # print("Dropped Features:", dropped_features)

    # model = xgb.XGBClassifier(**best_hyperparameters)
    # model.fit(X_selected, y)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"maybank/data/models/xgboost_model_{timestamp}.pkl"
    save_model(model, model_filename)

    model_config = {"models": {"xgboost": model_filename}}
    update_config_file("maybank/conf/base/models.yaml", model_config)
