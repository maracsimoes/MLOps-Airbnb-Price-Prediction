import pandas as pd
import logging
from typing import Dict, Tuple, Any, List

logger = logging.getLogger(__name__)

def model_predict(
    X: pd.DataFrame,
    model: Any,
    columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Gera predições usando o modelo carregado.

    Args:
        X (pd.DataFrame): DataFrame com dados de entrada (pré-processados).
        model: Modelo treinado (injetado pelo Kedro Data Catalog).
        columns (List[str]): Lista de colunas usadas pelo modelo.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - DataFrame com coluna 'y_pred' adicionada.
            - Dicionário com estatísticas descritivas do DataFrame resultante.
    """
    df = X.copy()
    y_pred = model.predict(df[columns])
    df["y_pred"] = y_pred

    stats = df.describe().to_dict()
    logger.info(f"Predicted {len(y_pred)} samples.")
    return df, stats
