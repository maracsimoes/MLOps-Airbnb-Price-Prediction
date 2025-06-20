import pandas as pd
import numpy as np
from mlops_project.pipelines.split_train_pipeline.nodes import split_data

def test_split_data_node():
    df = pd.DataFrame({
        "feature1": range(100),
        "feature2": np.random.rand(100),
        "target": [0, 1] * 50,
        "index": range(100)
    })

    X_train, X_test, y_train, y_test, best_columns = split_data(df)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert "index" not in X_train.columns

    assert all(col in df.columns for col in best_columns)
