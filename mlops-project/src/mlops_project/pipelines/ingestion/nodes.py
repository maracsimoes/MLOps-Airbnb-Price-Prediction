import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from typing import List   
from typing import Optional
from datetime import datetime

from great_expectations.core import ExpectationSuite, ExpectationConfiguration


from pathlib import Path

from kedro.config import OmegaConfigLoader
from kedro.framework.project import settings

conf_path = str(Path('') / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)
credentials = conf_loader["credentials"]



logger = logging.getLogger(__name__)

NUMERICAL_FEATURES: List[str] = [
    "host_total_listings_count",
    "accommodates",
    "bedrooms",
    "minimum_nights",
    "review_scores_rating",
    "host_days_active",
    "host_years_active",
    "amenities_length",
    "living_entertainment",
    "kitchen_dining",
    "bedroom",
    "bathroom",
    "baby_family",
    "laundry_cleaning",
    "safety_security",
    "outdoor_garden",
    "heating_cooling",
    "travel_access",
    "wellness_leisure",
    "guest_services",
    "misc_essentials",
]

CATEGORICAL_FEATURES: List[str] = [
    "neighbourhood",
    "property_type",
    "room_type",
]

BOOLEAN_FEATURES: List[str] = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

TARGET_COLUMN = "price"

def build_expectation_suite(expectation_suite_name: str, feature_group: str) -> ExpectationSuite:
    """
    Create a Great Expectations suite for one of three groups:
    * ``numerical_features``
    * ``categorical_features``
    * ``boolean_features``
    * ``target``

    Parameters
    ----------
    expectation_suite_name
        Name stored inside GX.
    feature_group
        One of the three strings above.

    Returns
    -------
    great_expectations.core.ExpectationSuite
    """
    expectation_suite_bank = ExpectationSuite(expectation_suite_name=expectation_suite_name)

    # Numerical features:
    if feature_group == "numerical_features":
        for col in NUMERICAL_FEATURES:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_in_type_list",
                    kwargs={"column": col, "type_list": ["float64", "int64", "Int64"]},
                )
            )

    # Categorical features:
    elif feature_group == "categorical_features":
        for col in CATEGORICAL_FEATURES:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_be_of_type",
                    kwargs={"column": col, "type_": "object"},
                )
            )

    # Boolean features:
    elif feature_group == "boolean_features":
        for col in BOOLEAN_FEATURES:
            expectation_suite_bank.add_expectation(
                ExpectationConfiguration(
                    expectation_type="expect_column_values_to_match_regex",
                    kwargs={"column": col, "regex": "^(True|False)$"},
                )
            )

    # Target feature (price)
    elif feature_group == "target":
        expectation_suite_bank.add_expectation(
            ExpectationConfiguration(
                "expect_column_values_to_be_in_type_list",
                kwargs={"column": TARGET_COLUMN, "type_list": ["float64", "int64"]},
            )
        )

    else:
        raise ValueError(
            "feature_group must be one of "
            "'numerical_features', 'categorical_features', "
            "'boolean_features', or 'target'"
        )

    logger.info(
        "Built expectation suite '%s' with %d expectations for group '%s'",
        expectation_suite_name,
        len(expectation_suite_bank.expectations),
        feature_group,
    )
    return expectation_suite_bank

import hopsworks

def to_feature_store(
    data: pd.DataFrame,
    group_name: str,
    feature_group_version: int,
    description: str,
    group_description: List[Dict[str, str]],
    validation_expectation_suite: ExpectationSuite,
    credentials_input: Dict[str, str],
    primary_key: List[str] | None = None,
    event_time: str | None = None,
) -> "hopsworks.feature_store_api.FeatureGroup":
    """
    Upload a DataFrame to Hopsworks Feature Store **after** validating it
    with a Great Expectations suite.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame you want in the feature store.
    group_name : str
        Name of the feature group in Hopsworks (e.g. ``"airbnb_listings"``).
    feature_group_version : int
        Numeric version of the feature group (1, 2, …).
    description : str
        Text description of the feature group.
    group_description : list[dict]
        One dict per feature, each like  
        ``{"name": "price", "description": "Nightly price in USD"}``
    validation_expectation_suite : ExpectationSuite
        Great Expectations suite used for data validation inside Hopsworks.
    credentials_input : dict
        Dict with keys ``"FS_PROJECT_NAME"`` and ``"FS_API_KEY"``.
    primary_key : list[str] | None, default None
        Primary‑key columns.  If ``None`` → the current DataFrame index.
    event_time : str | None, default None
        Name of event‑time column.  If ``None`` no event_time is set.

    Returns
    -------
    hopsworks.feature_store_api.FeatureGroup
        The FeatureGroup object created/updated in Hopsworks.
    """

    # 1. Connect to your Hopsworks project
    logger.info("Connecting to Hopsworks project '%s' ...",
                credentials_input["FS_PROJECT_NAME"])
    project = hopsworks.login(
        project=credentials_input["FS_PROJECT_NAME"],
        api_key_value=credentials_input["FS_API_KEY"],
    )
    feature_store = project.get_feature_store()

    # 2. Create (or fetch) the feature group
    object_feature_group = feature_store.get_or_create_feature_group(
        name=group_name,
        version=feature_group_version,
        description=description,
        primary_key=primary_key or ["index"],
        event_time=event_time,
        online_enabled=False,
        expectation_suite=validation_expectation_suite,
    )

    # 3. Insert the data
    logger.info("Inserting data into feature group '%s' v%s …",
                group_name, feature_group_version)
    object_feature_group.insert(
        features=data,
        overwrite=False,
        write_options={"wait_for_job": True},
    )

    # 4. Add human‑readable descriptions per feature
    for feat in group_description:
        object_feature_group.update_feature_description(feat["name"], feat["description"])

    # 5. Compute statistics 
    object_feature_group.statistics_config = {
        "enabled": True,
        "histograms": True,
        "correlations": True,
    }
    object_feature_group.update_statistics_config()
    object_feature_group.compute_statistics()

    logger.info("Feature group '%s' v%s ingested and validated.",
                group_name, feature_group_version)
    return object_feature_group

def ingestion(
    df1: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    df_full = df1

    target_col_config = parameters.get("target_col", {})
    target_column = target_col_config.get("target_column", "price")  # default "price"
    to_feature_store_flag = target_col_config.get("to_feature_store", False)
    
    # 2. Identify feature groups
    numerical_features = (
        df_full.select_dtypes(exclude=["object", "string", "category"])
        .columns
        .tolist()
    )
    categorical_features = [
        c for c in df_full.select_dtypes(include=["object", "string", "category"]).columns
        if c != target_column
    ]

    # 3. Add synthetic event‑time column
    months_int = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    df_full = df_full.reset_index()
    df_full["datetime"] = pd.to_datetime(
        {
            "year": datetime.now().year,
            "month": df_full["month"].map(months_int),
            "day": 1,
        }
    )

    # 4. Build GX expectation suites
    suite_num    = build_expectation_suite("numerical_expectations",   "numerical_features")
    suite_cat    = build_expectation_suite("categorical_expectations", "categorical_features")
    suite_target = build_expectation_suite("target_expectations",      "target")

    # 5. Slice DataFrames by group
    df_num    = df_full[["index", "datetime"] + numerical_features]
    df_cat    = df_full[["index", "datetime"] + categorical_features]
    df_target = df_full[["index", "datetime", target_column]]

    # 6. Optional: push to Hopsworks Feature Store
    if parameters.get("to_feature_store", False):
        logger.info("Uploading feature groups to Hopsworks …")

        to_feature_store(
            data=df_num,
            group_name="numerical_features",
            feature_group_version=1,
            description="Numerical engineered features from Airbnb listings",
            group_description=[{"name": col, "description": col} for col in numerical_features],
            validation_expectation_suite=suite_num,
            credentials_input=credentials["feature_store"],
            primary_key=["index"],
            event_time="datetime",
        )

        to_feature_store(
            data=df_cat,
            group_name="categorical_features",
            feature_group_version=1,
            description="Categorical engineered features from Airbnb listings",
            group_description=[{"name": col, "description": col} for col in categorical_features],
            validation_expectation_suite=suite_cat,
            credentials_input=credentials["feature_store"],
            primary_key=["index"],
            event_time="datetime",
        )
        
        to_feature_store(
            data=df_target,
            group_name="target_features",
            feature_group_version=1,
            description="Target (price) for Airbnb listings",
            group_description=[
                {"name": target_column, "description": "Nightly price USD"}
            ],
            validation_expectation_suite=suite_target,
            credentials_input=credentials["feature_store"],
            primary_key=["index"],
            event_time="datetime",
        )

    logger.info("Ingestion completed – final shape: %s", df_full.shape)
    return df_full