from kedro.pipeline import Pipeline, node, pipeline
from .nodes import feature_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection,
                inputs=["X_train_encoded", "y_train", "params:feature_selection"],
                outputs=["best_columns", "fitted_rfe"],
                name="model_feature_selection",
            ),
        ]
    )
