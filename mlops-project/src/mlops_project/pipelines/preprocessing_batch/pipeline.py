"""
This is a boilerplate pipeline 'preprocessing_batch'
"""
from kedro.pipeline import Pipeline, node

from .nodes import (change_data_types, clean_accommodates)


def create_pipeline(**_):
    return Pipeline(
        [
        node(change_data_types, "listings_raw", "listings_typedV3Batch", name="type_casting"),
        node(clean_accommodates, "listings_typedV3Batch", "preprocessed_batch_data", name="std_accommodates"),
        ]
    )
