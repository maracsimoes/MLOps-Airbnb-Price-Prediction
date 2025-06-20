from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import mlflow
import yaml
import numpy as np
import shap
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def model_train(X_train, X_test, y_train, y_test, parameters: dict, best_columns):
    with open('conf/local/mlflow.yml') as f:
        experiment_name = yaml.safe_load(f)['tracking']['experiment']['name']
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

    model_type = parameters.get("model_type", "linear_regression")

    if model_type == "gradient_boosting":
        model = GradientBoostingRegressor(**parameters.get('gradient_boosting_params', {}))
        logger.info("Using Gradient Boosting Regressor")
    elif model_type == "random_forest":
        model = RandomForestRegressor(**parameters.get('random_forest_params', {}))
        logger.info("Using Random Forest Regressor")
    else:
        model = LinearRegression()
        logger.info("Using Linear Regression")

    use_fs = parameters.get("use_feature_selection", {})
    if isinstance(use_fs, dict):
        apply_fs = use_fs.get(model_type, False)
    else:
        apply_fs = bool(use_fs)

    if apply_fs:
        X_train = X_train[best_columns]
        X_test = X_test[best_columns]

    y_train = np.ravel(y_train)

    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        results_dict = {
            'model': model.__class__.__name__,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
        }
        logger.info(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
        logger.info(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")

    if model_type in ["random_forest", "gradient_boosting"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train)
        shap.initjs()
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, show=False)
        return model, X_train.columns, results_dict, plt
    else:
        return model, X_train.columns, results_dict, None
