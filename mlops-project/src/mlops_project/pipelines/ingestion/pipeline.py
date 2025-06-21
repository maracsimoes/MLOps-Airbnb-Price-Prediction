from kedro.pipeline import node, pipeline
from .nodes import ingestion

def create_pipeline(**_) -> pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
                    "listings_raw",
                    "listings_raw_extra@optional",   #optional extra dataset
                    "parameters",
                ],
                outputs="ingested_data",
                name="ingestion",
            ),
        ]
    )