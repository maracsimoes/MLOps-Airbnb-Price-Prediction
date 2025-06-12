"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.13
"""


from kedro.pipeline import node, Pipeline
from .nodes import train_model_node

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_model_node,
                inputs=["X_train_scaled", "y_train_imputed"],
                outputs="trained_model",
                name="train_model_node"
            )
        ]
    )
