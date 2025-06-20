"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""
import great_expectations as gx
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from great_expectations.core.expectation_configuration import ExpectationConfiguration
import great_expectations as gx

from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

logger = logging.getLogger(__name__)


def get_validation_results(checkpoint_result):
    # validation_result is a dictionary containing one key-value pair
    validation_result_key, validation_result_data = next(iter(checkpoint_result["run_results"].items()))

    # Accessing the 'actions_results' from the validation_result_data
    validation_result_ = validation_result_data.get('validation_result', {})

    # Accessing the 'results' from the validation_result_data
    results = validation_result_["results"]
    meta = validation_result_["meta"]
    use_case = meta.get('expectation_suite_name')
    
    
    df_validation = pd.DataFrame({},columns=["Success","Expectation Type","Column","Column Pair","Max Value",\
                                       "Min Value","Element Count","Unexpected Count","Unexpected Percent","Value Set","Unexpected Value","Observed Value"])
    
    
    for result in results:
        # Process each result dictionary as needed
        success = result.get('success', '')
        expectation_type = result.get('expectation_config', {}).get('expectation_type', '')
        column = result.get('expectation_config', {}).get('kwargs', {}).get('column', '')
        column_A = result.get('expectation_config', {}).get('kwargs', {}).get('column_A', '')
        column_B = result.get('expectation_config', {}).get('kwargs', {}).get('column_B', '')
        value_set = result.get('expectation_config', {}).get('kwargs', {}).get('value_set', '')
        max_value = result.get('expectation_config', {}).get('kwargs', {}).get('max_value', '')
        min_value = result.get('expectation_config', {}).get('kwargs', {}).get('min_value', '')

        element_count = result.get('result', {}).get('element_count', '')
        unexpected_count = result.get('result', {}).get('unexpected_count', '')
        unexpected_percent = result.get('result', {}).get('unexpected_percent', '')
        observed_value = result.get('result', {}).get('observed_value', '')
        if type(observed_value) is list:
            #sometimes observed_vaue is not iterable
            unexpected_value = [item for item in observed_value if item not in value_set]
        else:
            unexpected_value=[]
        
        df_validation = pd.concat([df_validation, pd.DataFrame.from_dict( [{"Success" :success,"Expectation Type" :expectation_type,"Column" : column,"Column Pair" : (column_A,column_B),"Max Value" :max_value,\
                                           "Min Value" :min_value,"Element Count" :element_count,"Unexpected Count" :unexpected_count,"Unexpected Percent":unexpected_percent,\
                                                  "Value Set" : value_set,"Unexpected Value" :unexpected_value ,"Observed Value" :observed_value}])], ignore_index=True)
        
    return df_validation

def test_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the *raw* listings dataset with Great Expectations.

    1. Create / update a suite called ``ListingsRaw``.
    2. Run an in‑memory checkpoint.
    3. Fail Kedro run if critical assertions are violated.
    4. Return a DataFrame summarising each expectation.
    """
    # GX context 
    context = gx.get_context(context_root_dir="gx")

    # Register / update a pandas datasource
    datasource = context.sources.add_or_update_pandas("listings_ds")

    # Build or refresh expectation suite
    suite = context.add_or_update_expectation_suite("ListingsRaw")

    # a) Primary‑key column
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_not_be_null",
            kwargs={"column": "listing_id"},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_be_unique",
            kwargs={"column": "listing_id"},
        )
    )

    # b) Price & nights ranges
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_be_between",
            kwargs={"column": "price", "min_value": 1, "max_value": 500},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_be_between",
            kwargs={"column": "minimum_nights", "min_value": 1, "max_value": 40},
        )
    )
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_be_between",
            kwargs={"column": "maximum_nights", "min_value": 1, "max_value": 1150},
        )
    )

    # c) Bedrooms (nullable, but when present ≤ 10)
    suite.add_expectation(
        ExpectationConfiguration(
            "expect_column_values_to_be_between",
            kwargs={
                "column": "bedrooms",
                "min_value": 0,
                "max_value": 10,
                "ignore_row_if": "either_value_is_missing",
            },
        )
    )

    # d) Boolean flags only True / False
    for bcol in [
        "host_is_superhost",
        "host_has_profile_pic",
        "host_identity_verified",
        "instant_bookable",
    ]:
        if bcol in df.columns:
            suite.add_expectation(
                ExpectationConfiguration(
                    "expect_column_values_to_match_regex",
                    kwargs={"column": bcol, "regex": "^(True|False)$"},
                )
            )

    # Persist suite
    context.add_or_update_expectation_suite(suite)

    # Build batch & checkpoint
    asset   = datasource.add_dataframe_asset("in_memory", dataframe=df)
    request = asset.build_batch_request(dataframe=df)

    ckpt = gx.checkpoint.SimpleCheckpoint(
        name="ckpt_listings_raw",
        data_context=context,
        validations=[{"batch_request": request, "expectation_suite_name": "ListingsRaw"}],
    )
    ckpt_result = ckpt.run()

    # Flatten results for reporting
    validation_df = get_validation_results(ckpt_result)

    # Hard‑stop assertions on critical schema
    pd_ge = gx.from_pandas(df)
    assert pd_ge.expect_column_values_to_be_of_type("listing_id", "int64").success
    assert pd_ge.expect_column_values_to_be_of_type("price", "float64").success

    logger.info(" Listings_paris.csv passed Great Expectations unit tests.")
    return validation_df