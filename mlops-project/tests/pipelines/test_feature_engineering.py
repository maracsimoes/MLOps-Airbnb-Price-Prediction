import pandas as pd
import pytest
from src.mlops_project.pipelines.feature_engineering.nodes import feature_engineering_node, amenity_categories


from src.mlops_project.pipelines.feature_engineering.nodes import (
    fix_zero_total_listings,
    add_host_activity_columns_inplace,
    categorize_property_type_column,
    parse_amenities,
    add_amenities_features,
    add_standardized_amenities,
    add_amenity_category_counts,
    remove_irrelevant_columns,
    feature_engineering_node
)

def test_fix_zero_total_listings():
    df = pd.DataFrame({
        "host_id": [1, 1, 2, 3, 3, 3],
        "host_total_listings_count": [0, 0, 2, 0, 0, 0]
    })
    result = fix_zero_total_listings(df)
    expected = [2, 2, 2, 3, 3, 3]
    assert all(result["host_total_listings_count"] == expected)

def test_add_host_activity_columns_inplace():
    df = pd.DataFrame({
        "host_since": ["2020-01-01", "2021-01-01", "2019-06-01"]
    })
    add_host_activity_columns_inplace(df)
    max_date = pd.to_datetime(df["host_since"]).max()
    expected_days = (max_date - pd.to_datetime(df["host_since"])).dt.days
    expected_years = expected_days // 365
    assert all(df["host_days_active"] == expected_days)
    assert all(df["host_years_active"] == expected_years)

def test_categorize_property_type_column():
    df = pd.DataFrame({
        "property_type": ["Entire apartment", "Private room", "Shared room in hostel", "Unknown"]
    })
    result = categorize_property_type_column(df, inplace=False)
    assert result.loc[0, "property_type"] == "entire_place"
    assert result.loc[1, "property_type"] == "private_room"
    assert result.loc[2, "property_type"] == "shared_room"
    assert result.loc[3, "property_type"] == "Other"

def test_parse_amenities():
    s = '{"Wifi","TV"}'
    parsed = parse_amenities(s)
    assert "Wifi" in parsed and "TV" in parsed

    s = '["Wifi", "TV"]'
    parsed = parse_amenities(s)
    assert "Wifi" in parsed and "TV" in parsed

    s = None
    parsed = parse_amenities(s)
    assert parsed == []

def test_add_amenities_features():
    df = pd.DataFrame({
        "amenities": ['["Wifi", "TV", "Kitchen"]', '{"Wifi","TV"}', None]
    })
    result = add_amenities_features(df)
    assert result.loc[0, "amenities_length"] == 3
    assert result.loc[1, "amenities_length"] == 2
    assert result.loc[2, "amenities_length"] == 0

def test_add_standardized_amenities():
    df = pd.DataFrame({
        "amenities": ['["Wifi", "TV", "Kitchen"]', '{"Netflix", "Game Console"}', None]
    })
    result = add_standardized_amenities(df)
    assert "wifi" in result.loc[0, "standardized_amenities"]
    assert "netflix" in result.loc[1, "standardized_amenities"]
    assert result.loc[2, "standardized_amenities"] == []

def test_add_amenity_category_counts():
    df = pd.DataFrame({
        "standardized_amenities": [
            ["tv", "netflix", "wifi", "kitchen", "oven"],
            ["pool", "sauna", "hot tub"],
            []
        ]
    })
    result = add_amenity_category_counts(df, amenity_categories)
    assert result.loc[0, "living_entertainment"] >= 2  # tv, netflix
    assert result.loc[0, "kitchen_dining"] >= 2        # kitchen, oven
    assert result.loc[1, "wellness_leisure"] == 3
    assert result.loc[2, "living_entertainment"] == 0

def test_remove_irrelevant_columns():
    df = pd.DataFrame({
        "listing_id": [1],
        "host_id": [1],
        "host_since": ["2020-01-01"],
        "some_col": [123],
    })
    df_copy = df.copy()
    remove_irrelevant_columns(df)
    assert "listing_id" not in df.columns
    assert "host_id" not in df.columns
    assert "some_col" in df.columns

def test_feature_engineering_node_basic():
    df = pd.DataFrame({
        "host_id": [1, 1, 2],
        "host_total_listings_count": [0, 0, 1],
        "host_since": ["2020-01-01", "2020-01-01", "2019-01-01"],
        "property_type": ["Entire apartment", "Private room", "Shared room in hostel"],
        "amenities": ['["Wifi", "TV", "Kitchen"]', '{"Netflix", "Game Console"}', None],
        "listing_id": [101, 102, 103]
    })
    result = feature_engineering_node(df)
    assert "host_days_active" in result.columns
    assert "host_years_active" in result.columns
    assert "living_entertainment" in result.columns
    assert "listing_id" not in result.columns
    assert "host_id" not in result.columns
