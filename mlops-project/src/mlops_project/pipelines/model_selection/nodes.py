import pandas as pd
import logging
from typing import Dict, Any
import numpy as np
import yaml
import pickle
import warnings
warnings.filterwarnings("ignore", category=Warning)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import mlflow

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        logger.info(f"Experiment '{experiment_name}' not found. Creating new one.")
        return mlflow.create_experiment(experiment_name)
    return exp.experiment_id


def model_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    champion_dict: Dict[str, Any],
    champion_model: Any,
    parameters: Dict[str, Any],
) -> Any:
    """
    Compara vários modelos, faz tuning do melhor e retorna o novo champion.
    - champion_dict: dicionário com 'test_score' e 'regressor' do modelo atual.
    - champion_model: objeto do modelo atual (pickle).
    - parameters: contém grid de hyperparameters em parameters['hyperparameters'].
    """

    models_dict = {
        "RandomForestClassifier": RandomForestClassifier(),
        "GradientBoostingClassifier": GradientBoostingClassifier(),
    }
    initial_results = {}

    with open("conf/local/mlflow.yml") as f:
        mlflow_cfg = yaml.safe_load(f)
        experiment_name = mlflow_cfg["tracking"]["experiment"]["name"]
        experiment_id = _get_or_create_experiment_id(experiment_name)

    logger.info("Comparing model types")
    for model_name, model in models_dict.items():
        with mlflow.start_run(experiment_id=experiment_id, nested=True):
            mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
            y_train_arr = np.ravel(y_train)
            model.fit(X_train, y_train_arr)
            score = model.score(X_test, y_test)
            initial_results[model_name] = score
            run_id = mlflow.last_active_run().info.run_id
            logger.info(f"Logged {model_name} in run {run_id} with score {score}")

    best_name = max(initial_results, key=initial_results.get)
    best_model = models_dict[best_name]
    logger.info(f"Best base model: {best_name} ({initial_results[best_name]:.4f})")

    param_grid = parameters["hyperparameters"][best_name]
    logger.info(f"Tuning hyperparameters for {best_name}")
    with mlflow.start_run(experiment_id=experiment_id, nested=True):
        gs = GridSearchCV(best_model, param_grid, cv=2, scoring="accuracy", n_jobs=-1)
        gs.fit(X_train, np.ravel(y_train))
        best_model = gs.best_estimator_
        logger.info(f"GridSearch best score: {gs.best_score_:.4f}")

    pred_score = accuracy_score(y_test, best_model.predict(X_test))
    logger.info(f"Test accuracy of tuned model: {pred_score:.4f}")

    if champion_dict.get("test_score", -1) < pred_score:
        logger.info(f"New champion: {best_name} ({pred_score:.4f} > {champion_dict['test_score']:.4f})")
        return best_model
    else:
        logger.info(f"Champion remains: {champion_dict['regressor']} ({champion_dict['test_score']:.4f} ≥ {pred_score:.4f})")
        return champion_model
