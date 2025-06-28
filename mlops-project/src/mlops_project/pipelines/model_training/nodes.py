from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import mlflow
import yaml
import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import copy
logger = logging.getLogger(__name__)

def model_train(X_train, X_test, y_train, y_test, parameters: dict, best_columns):

    try:
        with open('conf/local/mlflow.yml') as f:
            experiment_name = yaml.safe_load(f)['tracking']['experiment']['name']
    except FileNotFoundError:
        experiment_name = "Default"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    model_types = parameters.get("model_type", ["linear_regression"])
    if isinstance(model_types, str):
        model_types = [model_types]

    use_fs = parameters.get("use_feature_selection", {})
    relevant_metric = parameters.get("relevant_metric", "test_rmse")

    best_score = None
    best_model = None
    best_results = None
    best_features = None
    best_plot = None

    for model_type in model_types:
        if model_type == "gradient_boosting":
            model = GradientBoostingRegressor(**parameters.get('gradient_boosting_params', {}))
            logger.info("Using Gradient Boosting Regressor")
        elif model_type == "random_forest":
            model = RandomForestRegressor(**parameters.get('random_forest_params', {}))
            logger.info("Using Random Forest Regressor")
        else:
            model = LinearRegression()
            logger.info("Using Linear Regression")

        apply_fs = use_fs.get(model_type, False) if isinstance(use_fs, dict) else bool(use_fs)
        X_train_used = X_train[best_columns] if apply_fs else X_train
        X_test_used = X_test[best_columns] if apply_fs else X_test

        y_train_flat = np.ravel(y_train)

        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            model.fit(X_train_used, y_train_flat)
            y_train_pred = model.predict(X_train_used)
            y_test_pred = model.predict(X_test_used)

            train_mae = mean_absolute_error(y_train_flat, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train_flat, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            results_dict = {
                'model': model.__class__.__name__,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
            }
            logger.info(f"{model_type} - Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
            logger.info(f"{model_type} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

        score = results_dict.get(relevant_metric, None)
        if score is not None and (best_score is None or score < best_score):
            best_score = score
            best_model = copy.deepcopy(model)
            best_results = results_dict
            best_features = X_train_used.columns
            if model_type in ["random_forest", "gradient_boosting"]:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_train_used)
                shap.initjs()
                plt.clf()
                shap.summary_plot(shap_values, X_train_used, feature_names=X_train_used.columns, show=False)
                best_plot = plt.gcf()
            else:
                best_plot = None

    return best_model, best_features, best_results, best_plot, best_features