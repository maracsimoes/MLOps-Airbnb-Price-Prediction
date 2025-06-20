import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.mlops_project.pipelines.model_predict.nodes import model_predict

def test_model_predict_basic():
    X = pd.DataFrame({
        "feature1": [1, 2, 3],
        "feature2": [4, 5, 6]
    })

    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0, 1, 0])

    columns = ["feature1", "feature2"]

    result_df, stats = model_predict(X, mock_model, columns)

    assert "y_pred" in result_df.columns
    assert len(result_df["y_pred"]) == len(X)
    assert all(result_df["y_pred"] == mock_model.predict.return_value)

    assert isinstance(stats, dict)
    assert "feature1" in stats
    assert "y_pred" in stats
    assert "mean" in stats["y_pred"]
