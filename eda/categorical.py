import pandas as pd


def build_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top N value counts for a categorical column."""
    value_counts = (
        df[column]
        .astype("string")
        .fillna("<missing>")
        .value_counts()
        .head(top_n)
        .reset_index()
    )

    value_counts.columns = [column, "count"]

    return value_counts