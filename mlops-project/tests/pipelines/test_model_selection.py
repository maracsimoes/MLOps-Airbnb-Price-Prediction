import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.mlops_project.pipelines.model_selection.nodes import model_selection

def test_model_selection_returns_model():
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8]
    })
    X_test = pd.DataFrame({
        "feature1": [2, 3],
        "feature2": [6, 7]
    })
    y_train = pd.DataFrame({"target": [0, 1, 0, 1]})
    y_test = pd.DataFrame({"target": [1, 0]})

    champion_dict = {"test_score": 0.5, "regressor": "RandomForestClassifier"}
    champion_model = MagicMock()
    parameters = {
        "hyperparameters": {
            "RandomForestClassifier": {"n_estimators": [10]},
            "GradientBoostingClassifier": {"n_estimators": [10]}
        }
    }

    with patch("src.mlops_project.pipelines.model_selection.nodes.mlflow") as mock_mlflow, \
         patch("src.mlops_project.pipelines.model_selection.nodes.GridSearchCV") as mock_gs, \
         patch("builtins.open"), \
         patch("yaml.safe_load", return_value={"tracking": {"experiment": {"name": "test_exp"}}}):

        mock_mlflow.get_experiment_by_name.return_value = None
        mock_mlflow.create_experiment.return_value = "1"
        mock_run = MagicMock()
        mock_run.info.run_id = "abc123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.last_active_run.return_value = mock_run

        mock_best_model = MagicMock()
        mock_best_model.predict.return_value = np.array([1, 0])
        mock_gs.return_value.fit.return_value = None
        mock_gs.return_value.best_estimator_ = mock_best_model
        mock_gs.return_value.best_score_ = 0.9

        result_model = model_selection(X_train, X_test, y_train, y_test, champion_dict, champion_model, parameters)

        assert result_model == mock_best_model or result_model == champion_model
