import pandas as pd
import numpy as np
from src.mlops_project.pipelines.feature_selection.nodes import feature_selection

def test_feature_selection_with_rfe(tmp_path):
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "feature3": [2, 3, 4, 5, 6],
    })
    y_train = pd.Series([0, 1, 0, 1, 0])

    parameters = {
        "feature_selection": "rfe",
        "baseline_model_params": {
            "n_estimators": 10,
            "random_state": 42
        }
    }

    selected = feature_selection(X_train, y_train, parameters)

    assert isinstance(selected, list)
    assert len(selected) <= X_train.shape[1]
    for col in selected:
        assert col in X_train.columns

def test_feature_selection_without_method():
    X_train = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6],
    })
    y_train = pd.Series([1, 0, 1])

    parameters = {}

    selected = feature_selection(X_train, y_train, parameters)

    assert selected == list(X_train.columns)
