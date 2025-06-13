"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from kedro.pipeline import node, Pipeline
from .nodes import feature_engineering



def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=feature_engineering,
            inputs="typed_data",
            outputs="features_data",
            name="feature_engineering_node",
        )
    ])
