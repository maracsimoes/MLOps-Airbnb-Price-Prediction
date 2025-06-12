"""
This is a boilerplate pipeline 'data_cleaning'
generated using Kedro 0.19.13
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import change_data_types, remove_irrelevant_columns


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=remove_irrelevant_columns,
            inputs="raw_data",
            outputs="filtered_data",
            name="remove_irrelevant_columns_node",
        ),
        node(
            func=change_data_types,
            inputs="filtered_data",
            outputs="typed_data",
            name="change_data_types_node",
        )
    ])