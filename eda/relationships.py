import pandas as pd


def calculate_pairwise_correlation(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    method: str = "pearson",
) -> float:
    """Calculate a pairwise correlation between two numeric columns."""
    return float(df[[x_column, y_column]].corr(method=method).iloc[0, 1])

