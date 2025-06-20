import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from src.mlops_project.pipelines.evaluation.nodes import evaluate_model_node

def test_evaluate_model(tmp_path, monkeypatch):
    X = pd.DataFrame({
        "feat1": np.random.rand(10),
        "feat2": np.random.rand(10)
    })
    y = X["feat1"] * 3 + X["feat2"] * 2 + np.random.normal(0, 0.1, size=10)

    model = LinearRegression()
    model.fit(X, y)

    monkeypatch.setattr("mlflow.log_figure", lambda fig, artifact_file: None)

    mae, rmse, shap_fig = evaluate_model_node(model, X, y)

    assert isinstance(mae, float)
    assert isinstance(rmse, float)
    assert mae >= 0
    assert rmse >= 0
    assert shap_fig is not None
