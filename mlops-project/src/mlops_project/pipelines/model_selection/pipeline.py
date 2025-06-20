from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_selection

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_selection,
                inputs=[
                    "X_train",               
                    "X_test",                 
                    "y_train",               
                    "y_test",                 
                    "val_metrics",           
                    "production_model",      
                    "params:model_selection",  
                ],
                outputs="champion_model",
                name="model_selection_node",
            ),
        ]
    )
