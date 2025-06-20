import pandas as pd
from src.mlops_project.pipelines.split_train_pipeline.nodes import split_data

def test_split_data():
    data = pd.DataFrame({
        "index": range(1, 11),
        "feature1": [1,2,3,4,5,6,7,8,9,10],
        "feature2": [10,20,30,40,50,60,70,80,90,100],
        "target": [0,1,0,1,0,1,0,1,0,1],
    })

    parameters = {
        "target_column": "target",
        "test_fraction": 0.2,
        "random_state": 42
    }

    X_train, X_test, y_train, y_test, best_columns = split_data(data, parameters)

    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None

    assert X_train.shape[0] == 8
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 8
    assert y_test.shape[0] == 2

    expected_cols = list(data.columns.drop(["target", "index"]))
    assert list(best_columns) == expected_cols
