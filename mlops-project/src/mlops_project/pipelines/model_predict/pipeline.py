from kedro.pipeline import Pipeline, node, pipeline
from .nodes import model_predict

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=model_predict,
                inputs=[
                    "preprocessed_batch_data",  
                    "production_model",         
                    "selected_features",        
                ],
                outputs=["df_with_predict", "predict_describe"],
                name="model_predict_node",
            ),
        ]
    )
