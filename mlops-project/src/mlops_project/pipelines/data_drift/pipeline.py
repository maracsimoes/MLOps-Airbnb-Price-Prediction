
"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_drift           # ← your node function

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=evaluate_drift,
                inputs=["X_test", "y_test", "y_pred"],  # <-- match catalog names
                outputs="regression_metrics",           # <-- catalog entry for the dict
                name="evaluate_drift",
            ),
        ]
    )