import pandas as pd


def get_constant_columns(df: pd.DataFrame) -> list[str]:
    """Return columns with one or fewer unique values, including missing values."""
    return [
        column
        for column in df.columns
        if df[column].nunique(dropna=False) <= 1
    ]


def get_high_cardinality_columns(
    column_profile: pd.DataFrame,
    unique_pct_threshold: float = 90,
    min_unique_count: int = 20,
) -> list[str]:
    """Return columns with a high proportion of unique values."""
    return column_profile[
        (column_profile["unique_pct"] > unique_pct_threshold)
        & (column_profile["unique_count"] > min_unique_count)
    ]["column"].tolist()


def get_possible_id_columns(
    column_profile: pd.DataFrame,
    unique_pct_threshold: float = 95,
) -> list[str]:
    """
    Return columns that may be identifiers.

    Heuristic:
    - very high uniqueness
    - no missing values
    """
    return column_profile[
        (column_profile["unique_pct"] > unique_pct_threshold)
        & (column_profile["missing_pct"] == 0)
    ]["column"].tolist()


def build_quality_checks(df: pd.DataFrame, column_profile: pd.DataFrame) -> pd.DataFrame:
    """Build a summary table of simple data quality checks."""
    constant_columns = get_constant_columns(df)
    high_cardinality_columns = get_high_cardinality_columns(column_profile)
    possible_id_columns = get_possible_id_columns(column_profile)

    return pd.DataFrame(
        {
            "check": [
                "Constant columns",
                "High-cardinality columns",
                "Possible ID columns",
            ],
            "count": [
                len(constant_columns),
                len(high_cardinality_columns),
                len(possible_id_columns),
            ],
            "columns": [
                ", ".join(constant_columns),
                ", ".join(high_cardinality_columns),
                ", ".join(possible_id_columns),
            ],
        }
    )