from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import (
    identify_data_types_node,
    split_data_node,
    handle_missing_values_node,
    handle_outliers_node,
    encode_categorical_node,
    scale_numeric_node
)

def create_pipeline(**kwargs):
    return Pipeline(
        [


            node(
                func=split_data_node,
                inputs=["features_data", "params:target_col"],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data_node"
            ),
            node(
                func=identify_data_types_node,
                inputs="X_train",
                outputs=["numerical", "categorical", "boolean", "datetime"],
                name="identify_data_types_node"
            ),

            node(
                func=handle_missing_values_node,
                inputs=["X_train", "X_val", "y_train", "y_val", "numerical", "categorical", "boolean"],
                outputs=["X_train_imputed", "X_val_imputed", "y_train_imputed", "y_val_imputed"],
                name="handle_missing_values_node"
            ),

            node(
                func=handle_outliers_node,
                inputs=["X_train_imputed", "numerical"],
                outputs="X_train_outliers_handled",
                name="handle_outliers_node"
            ),

            node(
                func=encode_categorical_node,
                inputs=["X_train_outliers_handled", "X_val_imputed"],
                outputs=["X_train_encoded", "X_val_encoded"],
                name="encode_categorical_node"
            ),

            node(
                func=scale_numeric_node,
                inputs=["X_train_encoded", "X_val_encoded"],
                outputs=["X_train_scaled", "X_val_scaled"],
                name="scale_numeric_node"
            ),
        ]
    )
