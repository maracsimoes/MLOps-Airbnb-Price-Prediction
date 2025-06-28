from kedro.pipeline import Pipeline, node
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_random,
                inputs="listings_raw",
                outputs=["ref_data", "ana_data"],
                name="split_random_node"
            ),
            node(
                func=le_fitting_node,
                inputs=["listings_raw"],
                outputs="label_encoder",
                name="le_fitting_node"
            )
        ]
    )
