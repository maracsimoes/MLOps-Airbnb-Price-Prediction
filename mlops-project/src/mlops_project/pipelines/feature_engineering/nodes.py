import pandas as pd
import numpy as np
import ast
from collections import Counter


def get_top_amenities(df: pd.DataFrame, top_n: int = 3) -> list:
    all_amenities = []
    for item in df['amenities'].dropna():
        try:
            all_amenities.extend(ast.literal_eval(item))
        except:
            continue
    top = [a for a, _ in Counter(all_amenities).most_common(top_n)]
    return top

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    top_amenities = get_top_amenities(df)
    for amenity in top_amenities:
        col_name = f'has_{amenity.lower().replace(" ", "_")}'
        df[col_name] = df['amenities'].apply(lambda x: amenity in ast.literal_eval(x) if pd.notnull(x) else False)
    df['n_amenities'] = df['amenities'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0)

    if 'host_since' in df.columns:
        last_date = df['host_since'].max()
        df['host_since_days'] = (last_date - df['host_since']).dt.days
        df = df.drop(columns=['host_since'])

    if 'price' in df.columns and 'bedrooms' in df.columns:
        df['price_per_bedroom'] = df.apply(
            lambda row: row['price'] / row['bedrooms'] if pd.notna(row['bedrooms']) and row['bedrooms'] > 0 else np.nan,
            axis=1
        )

    if 'host_is_superhost' in df.columns and 'number_of_reviews' in df.columns:
        df['superhost_x_reviews'] = df['host_is_superhost'].map({'t':1, 'f':0}) * df['number_of_reviews']

    if 'last_review' in df.columns:
        df['days_since_last_review'] = (df['last_review'].max() - df['last_review']).dt.days

    return df


