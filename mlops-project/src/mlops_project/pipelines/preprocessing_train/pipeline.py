
from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**_):
    return Pipeline(
        [
            
            node(
                preprocessing_node,
                inputs="ref_data",
                outputs="listings_preprocessed_train",
                name="preprocessing_train",
            ),

            node(
                feature_engineering_node,
                inputs="listings_preprocessed_train",
                outputs="listings_featured_train",
                name="feature_engineering_train",
            )
        ]
    )