"""
This is a boilerplate pipeline 'evaluation'
generated using Kedro 0.19.13
"""


from kedro.pipeline import node, Pipeline
from .nodes import evaluate_model_node

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=evaluate_model_node,
                inputs=["trained_model", "X_val_scaled", "y_val_imputed"],
                outputs=["val_mae", "val_rmse", "shap_figure"],  
                name="evaluate_model_node"
            )
        ])
