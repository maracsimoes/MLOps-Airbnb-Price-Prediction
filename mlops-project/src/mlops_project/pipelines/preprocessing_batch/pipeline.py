"""
This is a boilerplate pipeline 'preprocessing_batch'
"""
from kedro.pipeline import Pipeline, node

from .nodes import (change_data_types, clean_accommodates)


def create_pipeline(**_):
    return Pipeline(
        [
            
            node(
                change_data_types,
                inputs="listings_raw",
                outputs="listings_typed",
                name="cast_types",
            ),
            
            node(
                clean_accommodates, 
                "listings_pruned", 
                "listings_accomm", 
                name="fix_accommodates",
            )
        ]
    )