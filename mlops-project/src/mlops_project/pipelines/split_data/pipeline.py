from kedro.pipeline import Pipeline, node
from .nodes import split_random

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=split_random,
                inputs="ingested_data",
                outputs=["ref_data", "ana_data"],
                name="split_random_node"
            ),
        ]
    )
