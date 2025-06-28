from kedro.pipeline import node, pipeline
from .nodes import ingestion

def create_pipeline(**_) -> pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
                    "listings_raw",      
                    "params:target_col"],
                outputs="ingested_data",    
                name="ingestion",
            ),
        ]
    )