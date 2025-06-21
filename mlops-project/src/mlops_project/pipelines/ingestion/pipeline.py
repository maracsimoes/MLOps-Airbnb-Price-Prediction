from kedro.pipeline import Pipeline, node, pipeline

from .nodes import ingestion

def create_pipeline(**_) -> Pipeline:           
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
                    "listings_raw",        
                    "listings_raw_extra",       
                    "parameters"],
                outputs="ingested_data",    
                name="ingestion",
            ),
        ]
    )