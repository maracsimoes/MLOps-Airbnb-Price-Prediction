from kedro.pipeline import Pipeline, node
from mlops_project.pipelines.split_train_pipeline.nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_encode_scale_input,
                inputs=["listings_featured_train"],
                outputs=["X_train", "X_val", "y_train", "y_val", "scaler", "pre_train_columns", "input_medians", "input_modes"],
                name="split_encode_scale_node",
            )
        ]
    )
