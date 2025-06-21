
from kedro.pipeline import Pipeline, node

from .nodes import (change_data_types, clean_accommodates, keep_reasonable_bedroom_counts, keep_reasonable_prices, keep_reasonable_min_nights, keep_reasonable_max_nights)


def create_pipeline(**_):
    return Pipeline(
        [
            
            node(
                change_data_types,
                inputs="listings_raw",
                outputs="listings_typedV2train",
                name="cast_types",
            ),
            
            node(clean_accommodates, "listings_pruned", "listings_accomm", name="fix_accommodates"),
            node(
                keep_reasonable_bedroom_counts,
                "listings_accomm",
                "listings_bedrooms",
                name="filter_bedrooms",
            ),
            node(
                keep_reasonable_prices,
                "listings_bedrooms",
                "listings_price",
                name="filter_price",
            ),
            node(
                keep_reasonable_min_nights,
                "listings_price",
                "listings_min_nights",
                name="filter_min_nights",
            ),
            node(
                keep_reasonable_max_nights,
                "listings_min_nights",
                "listings_clean",
                name="filter_max_nights",
            ),
        ]
    )