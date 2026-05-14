import numpy as np
import pandas as pd


def calculate_correlation_matrix(
    df: pd.DataFrame,
    numeric_columns: list[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Calculate a correlation matrix for numeric columns."""
    return df[numeric_columns].corr(method=method)


def build_correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    """Convert a correlation matrix into sorted unique column pairs."""
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )

    corr_pairs.columns = ["column_1", "column_2", "correlation"]
    corr_pairs["absolute_correlation"] = corr_pairs["correlation"].abs()

    return corr_pairs.sort_values(
        by="absolute_correlation",
        ascending=False,
    )