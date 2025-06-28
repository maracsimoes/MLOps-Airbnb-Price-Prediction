
from kedro.pipeline import Pipeline, node

from .nodes import *


def create_pipeline(**_):
    return Pipeline(
        [
            node(
                preprocessing_node,
                inputs="ana_data",
                outputs="listings_preprocessed_batch",
                name="preprocessing_batch",
            ),
            node(
                feature_engineering_node,
                inputs="listings_preprocessed_batch",
                outputs="listings_featured_batch",
                name="feature_engineering_batch",
            ),
            node(
                impute_batch_data_node,
                inputs=["listings_featured_batch", "input_medians", "input_modes"],
                outputs="batch_ready_data",
                name="impute_batch_data",
            ),
            node(
                encode_and_scale_node,
                inputs=["batch_ready_data"],
                outputs="preprocessed_batch_data",
                name="encode_and_scale_batch_data",
            ),
        ]
    )