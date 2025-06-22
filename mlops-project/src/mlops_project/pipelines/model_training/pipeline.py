from kedro.pipeline import Pipeline, node
from .nodes import model_train

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=model_train,
                inputs=["X_train_encoded", "X_val_encoded", "y_train", "y_val", "params:model_training", "best_columns"],
                outputs=["production_model", "selected_features", "metrics", "shap_plot"],
                name="model_training_node",
            ),
        ]
    )
