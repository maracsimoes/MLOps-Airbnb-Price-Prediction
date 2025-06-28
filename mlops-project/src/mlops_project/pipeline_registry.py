from __future__ import annotations
from kedro.pipeline import Pipeline

from mlops_project.pipelines import (
    data_unit_tests,
    evaluation,
    feature_selection,
    model_predict,
    model_training,
    preprocessing_train,
    preprocessing_batch,
    split_data,
    split_train_pipeline,
    ingestion
)


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "ingestion": ingestion.create_pipeline(), 
        #"data_unit_tests": data_unit_tests.create_pipeline(),
        #"evaluation": evaluation.create_pipeline(),
        "feature_selection": feature_selection.create_pipeline(),
        "model_predict": model_predict.create_pipeline(),
        "model_training": model_training.create_pipeline(),
        "preprocessing": preprocessing_train.create_pipeline(),
        "preprocessing_batch": preprocessing_batch.create_pipeline(),
        "split_data": split_data.create_pipeline(),
        "split_train_pipeline": split_train_pipeline.create_pipeline(),
        "__default__": Pipeline([
            #ingestion.create_pipeline(),
            #data_unit_tests.create_pipeline(),
            split_data.create_pipeline(),
            preprocessing_train.create_pipeline(),
            split_train_pipeline.create_pipeline(),
            feature_selection.create_pipeline(),
            model_training.create_pipeline(),
            preprocessing_batch.create_pipeline(),
            model_predict.create_pipeline(),
            #evaluation.create_pipeline()
        ])
    }
