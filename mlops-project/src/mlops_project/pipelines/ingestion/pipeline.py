from kedro.pipeline import node, pipeline
from .nodes import ingestion

def create_pipeline(**_) -> pipeline:
    return pipeline(
        [
            node(
                func=ingestion,
                inputs=[
<<<<<<< HEAD
                    "listings_raw",        
                    "listings_raw_extra",       
                    "params:target_col"],
                outputs="ingested_data",    
=======
                    "listings_raw",
                    "listings_raw_extra@optional",   #optional extra dataset
                    "parameters",
                ],
                outputs="ingested_data",
>>>>>>> a135b283697058b5eee1160f3d9709d3161aa8b5
                name="ingestion",
            ),
        ]
    )