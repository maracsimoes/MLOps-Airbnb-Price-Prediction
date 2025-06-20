import pandas as pd
from typing import Tuple

def split_random(
    df: pd.DataFrame,
    frac: float = 0.8,
    random_state: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Faz split aleatório do dataframe em dois subsets.

    Args:
        df: dataframe original
        frac: fração para treino (default 0.8)
        random_state: semente para reprodutibilidade (default 200)

    Returns:
        ref_data: dataframe com frac de dados (exemplo treino)
        ana_data: restante (exemplo teste)
    """
    ref_data = df.sample(frac=frac, random_state=random_state)
    ana_data = df.drop(ref_data.index)

    return ref_data, ana_data
