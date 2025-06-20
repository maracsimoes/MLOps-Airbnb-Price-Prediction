import pandas as pd
import numpy as np
from src.mlops_project.pipelines.model_training.nodes import model_train

def test_model_train():
    X_train = pd.DataFrame({
        "feat1": np.random.rand(10),
        "feat2": np.random.rand(10),
    })
    y_train = pd.Series(np.random.randint(0, 2, 10))

    X_test = pd.DataFrame({
        "feat1": np.random.rand(4),
        "feat2": np.random.rand(4),
    })
    y_test = pd.Series(np.random.randint(0, 2, 4))

    parameters = {
        "model_type": "linear_regression",
        "use_feature_selection": False
    }
    best_columns = X_train.columns.tolist()

    model, selected_features, metrics, shap_plot = model_train(
        X_train, X_test, y_train, y_test, parameters, best_columns
    )

    assert metrics is not None
    assert "train_mae" in metrics or "train_accuracy" in metrics or "train_score" in metrics
    assert "test_mae" in metrics or "test_accuracy" in metrics or "test_score" in metrics
    assert list(selected_features) == list(best_columns) or isinstance(selected_features, list)
