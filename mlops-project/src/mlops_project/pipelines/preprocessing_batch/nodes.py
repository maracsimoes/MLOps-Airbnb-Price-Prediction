"""
Nodes for the `preprocessing_train` pipeline
"""

from __future__ import annotations

import os
import json
import re
from typing import List, Iterable, Any
import logging
from typing import List, Tuple
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

BOOL_COLS: List[str] = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

DATE_COLS: List[str] = ["host_since"]


def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast raw columns to proper dtypes.

    • 't'/'f'/'True'/'False' → boolean  
    • 'host_since'          → datetime64[ns]  
    • 'bedrooms'            → pandas nullable Int64
    """
    out = df.copy()
    out['district'] = out['district'].astype(str)
    # Booleans
    for col in set(BOOL_COLS) & set(out.columns):
        out[col] = (
            out[col]
            .astype(str)
            .str.lower()
            .map({"t": True, "f": False, "true": True, "false": False})
            .astype("boolean")
        )

    # Dates
    for col in set(DATE_COLS) & set(out.columns):
        out[col] = pd.to_datetime(out[col], errors="coerce")

    # Nullable integer
    if "bedrooms" in out.columns:
        out["bedrooms"] = pd.to_numeric(out["bedrooms"], errors="coerce").astype("Int64")

    logger.debug("change_data_types(): dtypes after cast\n%s", out.dtypes)
    return out

def replace_zeros_with_nan(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Return a copy where 0s in *column* are replaced by NaN.
    """
    out = df.copy()
    n_before = out[column].eq(0).sum()
    out[column] = out[column].replace(0, np.nan)
    logger.debug("replace_zeros_with_nan(): replaced %d zeros in '%s'", n_before, column)
    return out


def filter_by_max(
    df: pd.DataFrame, column: str, max_value: float | int
) -> pd.DataFrame:
    """
    Keep rows where *column* ≤ *max_value* (NaNs are kept).

    Returns a new DataFrame.
    """
    mask = df[column].le(max_value) | df[column].isna()
    logger.debug(
        "filter_by_max(): removed %d rows where %s > %s",
        (~mask).sum(), column, max_value
    )
    return df[mask].copy()


def filter_by_range(
    df: pd.DataFrame,
    column: str,
    min_exclusive: float | int,
    max_inclusive: float | int,
) -> pd.DataFrame:
    """
    Keep rows where *column* is > *min_exclusive* and ≤ *max_inclusive*.

    NaNs are dropped.
    """
    mask = (df[column] > min_exclusive) & (df[column] <= max_inclusive)
    logger.debug(
        "filter_by_range(): removed %d rows outside (%s, %s] in '%s'",
        (~mask).sum(), min_exclusive, max_inclusive, column
    )
    return df[mask].copy()

def clean_accommodates(df: pd.DataFrame) -> pd.DataFrame:
    """Replace 0 → NaN in 'accommodates'."""
    return replace_zeros_with_nan(df, column="accommodates")


def keep_reasonable_bedroom_counts(df: pd.DataFrame, max_bedrooms: int = 10) -> pd.DataFrame:
    """Drop listings with > *max_bedrooms* bedrooms (keep NaNs)."""
    return filter_by_max(df, column="bedrooms", max_value=max_bedrooms)


def keep_reasonable_prices(df: pd.DataFrame,
                           min_price: float = 0,
                           max_price: float = 500) -> pd.DataFrame:
    """Keep listings where 0 < price ≤ 500 (by default)."""
    return filter_by_range(df, "price", min_price, max_price)


def keep_reasonable_min_nights(df: pd.DataFrame, max_nights: int = 40) -> pd.DataFrame:
    """Drop rows with minimum_nights > max_nights."""
    return filter_by_max(df, "minimum_nights", max_nights)


def keep_reasonable_max_nights(df: pd.DataFrame, max_nights: int = 1150) -> pd.DataFrame:
    """Drop rows with maximum_nights > max_nights."""
    return filter_by_max(df, "maximum_nights", max_nights)

def preprocessing_node(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply a series of preprocessing steps to the input DataFrame.

    This includes:
    - Changing data types
    - Cleaning 'accommodates' column
    - Filtering by bedroom count, price, and nights

    Returns a preprocessed DataFrame.
    """
    df = change_data_types(df)
    df = clean_accommodates(df)
    df = keep_reasonable_bedroom_counts(df)
    df = keep_reasonable_prices(df)
    df = keep_reasonable_min_nights(df)
    df = keep_reasonable_max_nights(df)
    
    return df


def fix_zero_total_listings(
    df: pd.DataFrame,
    host_id_col: str = "host_id",
    total_col: str = "host_total_listings_count",
) -> pd.DataFrame:
    """
    Replace *host_total_listings_count* values that are **0** by the
    actual number of listings each host owns (computed within the DataFrame).

    Parameters
    ----------
    df : pd.DataFrame
        Input listings table.
    host_id_col : str, default "host_id"
        Name of the host‑identifier column.
    total_col : str, default "host_total_listings_count"
        Column holding the host’s declared total listing count.

    Returns
    -------
    pd.DataFrame
        A copy of *df* where any 0s in *total_col* have been replaced by
        the host’s real listing count.  (Hosts that already had
        a positive number remain unchanged.)
    """
    df = df.copy()

    actual_counts = (
        df.groupby(host_id_col)[host_id_col]
        .size()
        .rename("actual_listings")
    )

    df = df.join(actual_counts, on=host_id_col)

    df[total_col] = np.where(
        df[total_col] == 0,
        df["actual_listings"],
        df[total_col],
    )

    df = df.drop(columns="actual_listings")

    return df


def add_host_activity_columns_inplace(
    df: pd.DataFrame,
    since_col: str = "host_since",
    days_col: str = "host_days_active",
    years_col: str = "host_years_active",
) -> None:
    """
    Add two columns measuring host activity duration **in-place**:
    - *days_col*: days between max date in *since_col* and each host's date.
    - *years_col*: integer years active (days // 365).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to modify.
    since_col : str, default "host_since"
        Column with host start dates.
    days_col : str, default "host_days_active"
        Name for new days active column.
    years_col : str, default "host_years_active"
        Name for new years active column.

    Returns
    -------
    None
        Modifies *df* in place.
    """
    df[since_col] = pd.to_datetime(df[since_col], errors="coerce")
    max_date = df[since_col].max()
    df[days_col] = (max_date - df[since_col]).dt.days
    df[years_col] = (df[days_col] // 365).astype(int)

def categorize_property_type_column(
    df: pd.DataFrame,
    col: str = "property_type",
    groups: dict | None = None,
    default: str = "Other",
    inplace: bool = True
) -> pd.DataFrame:
    """
    Replace raw property_type values with their group/category names.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the property type column.
    col : str, default "property_type"
        Name of the column to categorize.
    groups : dict, optional
        Dictionary mapping group names to lists of raw property types.
        Uses default Airbnb groups if None.
    default : str, default "Other"
        Category name assigned to unknown property types.
    inplace : bool, default True
        If True, modify df in place and return it.
        Otherwise, return a new DataFrame with modified column.

    Returns
    -------
    pd.DataFrame
        DataFrame with the property_type column replaced by group labels.
    """

    if groups is None:
        groups = {
            "entire_place": [
                "Entire apartment", "Entire house", "Entire guesthouse",
                "Entire guest suite", "Entire place", "Entire bed and breakfast",
                "Entire townhouse", "Entire villa", "Entire home/apt",
                "Entire serviced apartment", "Entire condominium",
                "Entire loft", "Entire cottage", "Entire bungalow",
                "Entire floor", "Entire chalet", "Casa particular"
            ],
            "private_room": [
                "Private room in loft", "Private room in boat",
                "Private room in condominium", "Private room in chalet",
                "Private room in guest suite", "Private room in bed and breakfast",
                "Private room in cabin", "Private room in serviced apartment",
                "Private room in apartment", "Private room",
                "Private room in house", "Private room in villa",
                "Private room in guesthouse", "Private room in townhouse",
                "Private room in earth house", "Private room in hostel",
                "Private room in casa particular", "Private room in nature lodge",
                "Private room in houseboat", "Room in serviced apartment",
                "Room in bed and breakfast"
            ],
            "shared_room": [
                "Shared room in hostel", "Shared room in condominium",
                "Shared room in serviced apartment", "Shared room in apartment",
                "Shared room in cabin", "Shared room in loft",
                "Shared room in bed and breakfast", "Shared room in tiny house",
                "Shared room in house", "Shared room in igloo",
                "Shared room in guest suite", "Shared room in guesthouse",
                "Shared room in townhouse", "Shared room in boutique hotel"
            ],
            "unique_stays": [
                "Cave", "Tiny house", "Boat", "Earth house", "Campsite",
                "Houseboat", "Dome house", "Camper/RV", "Treehouse",
                "Island", "Barn"
            ],
            "hotel": [
                "Room in boutique hotel", "Room in aparthotel",
                "Room in hotel", "Room in hostel"
            ]
        }

 # Normalize group keys
    lookup = {raw.lower(): grp for grp, raw_list in groups.items() for raw in raw_list}

    # Normalize dataframe column
    normalized = df[col].astype(str).str.lower().str.strip()

    new_col = normalized.map(lookup).fillna(default)

    if inplace:
        df[col] = new_col
        return df
    else:
        return df.assign(**{col: new_col})
    
def parse_amenities(s):
    if pd.isna(s):
        return []
    # 1) try strict JSON first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # 2) fall back to common Airbnb quirks
        #    e.g. "{Wifi,\"TV\"}"  or  "['Wifi','TV']"
        s_fixed = (
            s.replace("'", '"')        # single → double quotes
             .strip("{}")              # drop outer curly braces
        )
        try:
            return json.loads(f'[{s_fixed}]')
        except Exception:
            return []
        

RAW_KEYWORDS = [
    "kettle", "dedicated workspace table", "indoor fireplace", "coffee machine",
    "first aid kit", "stove", "washerin unit", "kitchen", "single level home",
    "sound system", "clothing storage", "refrigerator", "garden", "oven", "hdtv",
    "netflix", "smoking allowed", "tv", "body soap", "outdoor furniture", "crib",
    "coffee", "movie projector", "babysitter recommendations", "host greets you",
    "baby safety gates", "pool table", "air conditioning", "parking", "outlet covers",
    "heating", "window guards", "waterfront", "lake access", "bedroom comforts",
    "shampoo", "washer", "fireplace guards", "baby monitor", "conditioner", "ac unit",
    "extra pillows and blankets", "childrens dinnerware", "hot water", "iron", "dryer",
    "childrens books and toys", "wifi", "game console", "barbecue", "shower gel",
    "breakfast", "baking sheet", "cleaning before checkout", "patio", "balcony", "pool",
    "self checkin", "lockbox", "sauna", "keypad", "hangers", "drying rack for clothing",
    "fans", "wine glasses", "private entrance", "dryerin", "bathtub",
    "dishes and silverware", "microwave", "outdoor shower", "heater", "high chair",
    "safe", "beachfront", "smoke alarm", "pets allowed", "fan", "ethernet", "dining table",
    "dedicated workspace", "elevator", "laundromat", "gym", "table corner guards",
    "toaster", "cleaning products", "hot tub", "nespresso machine", "ev charger",
    "smart lock", "record player", "fridge", "outdoor dining area", "rice maker",
    "essentials", "luggage dropoff allowed", "carbon monoxide alarm",
    "long term stays allowed", "bathroom essentials", "dishwasher", "freezer", "bidet",
    "skiinskiout", "building staff", "washer in", "roomdarkening shades", "mosquito net",
    "fire extinguisher", "beach essentials", "garden or backyard", "changing table",
    "cooking basics", "piano", "bed linens", "bbq grill", "baby bath", "bread maker",
    "electric blinds", "rain shower", "bikes", "wine cellar", "gated pproperty",
    "wine cooler", "ice machine", "board games", "suitable for events",
    "decorative fireplace", "trash compactor", "espresso machine", "dual vanity",
    "wet bar", "private living room", "library", "massage bed", "office", "music system",
    "dvd player", "outdoor seating", "internet", "gated community", "video games",
    "sun loungers", "spa", "ipod dock", "game room", "steam room", "projector",
    "dining area", "books", "ironing board", "lounge area", "blender", "restaurant",
    "gas grill", "smoking parlor", "terrace", "woodburning fireplace", "bluray player",
    "lock on bedroom door", "ping pong table", "garage", "ipad", "room service",
    "housekeeping", "toiletries", "bathrobes", "bed sheets and pillows", "selfparking",
    "alarm system", "airport shuttle", "fitness center", "minibar", "bar",
    "laundry services", "courtyard", "concierge", "slippers", "hammam",
    "bluetooth speaker", "printer", "turndown service", "bottled water", "linens",
    "desk", "kitchenette", "security cameras"
]

def add_amenities_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns derived from 'amenities':
    - 'amenities_array': parsed list of amenities
    - 'amenities_length': count of amenities

    Returns a copy of the DataFrame with new columns added.
    """
    df = df.copy()
    df['amenities_array'] = df['amenities'].apply(parse_amenities)
    df['amenities_length'] = df['amenities_array'].str.len()
    return df

DEFAULT_KEYWORDS = sorted({
    re.sub(r"[^a-zA-Z\s]", "", kw.lower()).strip()
    for kw in RAW_KEYWORDS
})

def _parse_amenities(raw: Any) -> List[str]:
    if pd.isna(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    try:
        fixed = raw.replace("'", '"').strip("{}")
        return json.loads(f"[{fixed}]")
    except Exception:
        return []

def add_standardized_amenities(
    df: pd.DataFrame,
    amenity_col: str = "amenities",
    output_col: str = "standardized_amenities",
    keywords: Iterable[str] | None = None,
    inplace: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    keyword_set = {
        re.sub(r"[^a-zA-Z\s]", "", kw.lower()).strip()
        for kw in (keywords or DEFAULT_KEYWORDS)
    }

    def _match_keywords(amenities: List[str]) -> List[str]:
        matches = set()
        for amenity in amenities:
            norm = re.sub(r"[^a-zA-Z\s]", "", amenity.lower()).strip()
            for kw in keyword_set:
                if kw in norm:
                    matches.add(kw)
                    break
        return sorted(matches)

    parsed = df[amenity_col].apply(_parse_amenities)
    standardized = parsed.apply(_match_keywords)
    if inplace:
        df[output_col] = standardized
        return df
    else:
        return df.assign(**{output_col: standardized})
    
amenity_categories = {
    "living_entertainment": [
        "tv", "hdtv", "netflix", "sound system", "movie projector", "record player",
        "dvd player", "bluray player", "ipod dock", "music system", "books",
        "board games", "game console", "video games", "ping pong table",
        "game room", "lounge area", "projector"
    ],
    "kitchen_dining": [
        "kitchen", "microwave", "oven", "stove", "toaster", "fridge", "refrigerator",
        "freezer", "rice maker", "bread maker", "nespresso machine", "coffee machine",
        "espresso machine", "kettle", "cooking basics", "dishes and silverware",
        "baking sheet", "blender", "dining table", "dining area", "wine glasses",
        "wine cooler", "wine cellar", "gas grill", "bbq grill", "barbecue", "dishwasher",
        "kitchenette", "ice machine"
    ],
    "bedroom": [
        "bed linens", "bedroom comforts", "bed sheets and pillows", "clothing storage",
        "extra pillows and blankets", "hangers", "ironing board", "iron",
        "roomdarkening shades", "linens", "alarm system"
    ],
    "bathroom": [
        "body soap", "shampoo", "conditioner", "shower gel", "bathtub", "bidet",
        "hot water", "bathroom essentials", "toiletries", "dual vanity", "rain shower",
        "bathrobes"
    ],
    "baby_family": [
        "crib", "high chair", "baby bath", "baby monitor", "baby safety gates",
        "childrens books and toys", "childrens dinnerware", "changing table",
        "table corner guards"
    ],
    "laundry_cleaning": [
        "washer", "washerin unit", "washer in", "dryer", "dryerin",
        "drying rack for clothing", "laundromat", "cleaning products",
        "cleaning before checkout", "laundry services"
    ],
    "safety_security": [
        "smoke alarm", "carbon monoxide alarm", "fire extinguisher",
        "fireplace guards", "outlet covers", "window guards", "mosquito net",
        "security cameras", "smart lock", "lockbox", "keypad",
        "lock on bedroom door", "alarm system"
    ],
    "outdoor_garden": [
        "patio", "balcony", "terrace", "garden", "garden or backyard",
        "outdoor furniture", "outdoor dining area", "outdoor seating",
        "outdoor shower", "sun loungers", "beachfront", "lake access",
        "beach essentials", "bikes", "gated community", "gated pproperty", "hammock"
    ],
    "heating_cooling": [
        "air conditioning", "ac unit", "fan", "fans", "heater", "heating",
        "indoor fireplace", "woodburning fireplace", "decorative fireplace"
    ],
    "travel_access": [
        "self checkin", "luggage dropoff allowed", "long term stays allowed",
        "private entrance", "suitable for events", "garage", "parking",
        "selfparking", "airport shuttle"
    ],
    "wellness_leisure": [
        "pool", "sauna", "hot tub", "hammam", "massage bed", "spa", "steam room"
    ],
    "workspace_tech": [
        "dedicated workspace", "dedicated workspace table", "desk", "office",
        "ethernet", "printer", "ipad", "internet", "wifi"
    ],
    "guest_services": [
        "host greets you", "room service", "housekeeping", "turndown service",
        "concierge", "minibar", "bar", "restaurant", "bottled water", "slippers",
        "building staff", "books", "library", "courtyard"
    ],
    "misc_essentials": [
        "essentials", "first aid kit", "pets allowed", "outlet covers"
    ]
}

def add_amenity_category_counts(df, categories_dict, amenities_col='standardized_amenities'):
    """
    Add one column per category to df with counts of matching keywords in the amenities list.

    Parameters:
    - df: pd.DataFrame
    - categories_dict: dict[str, list[str]] — keys = category names, values = list of keywords
    - amenities_col: str — column name in df with list of standardized amenities

    Returns:
    - pd.DataFrame with added category count columns (modifies df in place)
    """

    def count_category_matches(amenities, keywords):
        if not amenities:
            return 0
        return sum(1 for k in keywords if k in amenities)

    for category, keywords in categories_dict.items():
        df[category] = df[amenities_col].apply(lambda a: count_category_matches(a, keywords))

    return df

def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop predefined irrelevant columns from the DataFrame.

    Parameters:
    - df: pd.DataFrame — input DataFrame
    - inplace: bool — if True, modify df in place; if False, return new DataFrame

    Returns:
    - None if inplace=True (modifies df)
    - new DataFrame if inplace=False
    """
    cols_to_drop=['listing_id', 'host_id', 'host_since', 'host_response_rate', 'host_acceptance_rate', 'latitude', 'longitude', 'maximum_nights', 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'name', 'host_location', 'city', 'district', 'host_response_time', 'amenities', 'amenities_array', 'standardized_amenities']

    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def feature_engineering_node(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline node function.
    """
    df = preprocessing_node(df)
    df = fix_zero_total_listings(df)
    add_host_activity_columns_inplace(df)
    df = categorize_property_type_column(df, inplace=True)
    df = add_amenities_features(df)
    df = add_standardized_amenities(df, inplace=True)
    df = add_amenity_category_counts(df, amenity_categories)
    df = remove_irrelevant_columns(df)
    return df

def encode_and_scale_node(X):
    # Encode categorical columns
    with open(os.path.join(os.getcwd(), 'data', '04_feature', 'le_encoder.pkl'), 'rb') as f:
                le_dict = pickle.load(f)
    with open(os.path.join(os.getcwd(), 'data', '04_feature', 'scaler.pkl'), 'rb') as s:
                scaler = pickle.load(s)
    for col, le in le_dict.items():
        if col in X.columns:
            X[col] = le.transform(X[col].astype(str)).astype(int)


    # --- drop the target if it slipped in ----------------
    X = X.drop(columns=["price"], errors="ignore")
    # Apply scaling to numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    X[numeric_cols] = scaler.transform(X[numeric_cols])
    return X

def impute_batch_data_node(df, medians, modes):
    """
    Impute missing values in batch data using medians and modes from training.

    Args:
        df (pd.DataFrame): Batch data to impute.
        medians (dict): {column: median_value} for numerical features.
        modes (dict): {column: mode_value} for categorical/boolean features.

    Returns:
        pd.DataFrame: Imputed batch data.
    """
    # Impute numerical columns
    for col, median_val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    # Impute categorical/boolean columns
    for col, mode_val in modes.items():
        if col in df.columns:
            df[col] = df[col].fillna(mode_val)
    return df

__all__ = [
    "change_data_types",
    "remove_irrelevant_columns",
    "clean_accommodates",
    "keep_reasonable_bedroom_counts",
    "keep_reasonable_prices",
    "keep_reasonable_min_nights",
    "keep_reasonable_max_nights",
    "fix_zero_total_listings",
    "add_host_activity_columns_inplace",
    "categorize_property_type_column",
    "add_amenities_features",
    "add_standardized_amenities",
    "add_amenity_category_counts",
    "remove_irrelevant_columns",
    "feature_engineering_node",
    "preprocessing_node",
    "impute_batch_data_node",
    "encode_and_scale_node",
]


