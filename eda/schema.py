import numpy as np
import pandas as pd


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns pandas recognises as numeric."""
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """
    Return likely categorical columns.

    For now, this includes object, category, and boolean columns.
    Later we can improve this with semantic detection for IDs, dates, codes, etc.
    """
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def is_numeric_column(column: str, numeric_columns: list[str]) -> bool:
    """Check whether a column is in the numeric column list."""
    return column in numeric_columns