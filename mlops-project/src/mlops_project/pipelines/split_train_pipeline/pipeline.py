from kedro.pipeline import Pipeline, node
from mlops_project.pipelines.split_train_pipeline.nodes import split_data

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_training_data"],
                outputs=["X_train", "X_val", "y_train", "y_val", "best_columns"],
                name="split_train_node",
            ),
        ]
    )
