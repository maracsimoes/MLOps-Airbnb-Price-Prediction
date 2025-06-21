from typing import Dict
from kedro.pipeline import Pipeline
from mlops_project.pipelines.preprocessing_batch import create_pipeline as create_preprocessing_batch_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "preprocessing_batch": create_preprocessing_batch_pipeline(),
        "__default__": create_preprocessing_batch_pipeline(),
    }
