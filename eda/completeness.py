import pandas as pd


def build_missing_summary(column_profile: pd.DataFrame) -> pd.DataFrame:
    """Return missingness summary from the column profile."""
    return column_profile[
        ["column", "missing_count", "missing_pct", "non_null_count"]
    ].copy()


def build_row_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return row-level missingness summary."""
    row_missing = df.isna().sum(axis=1)

    return pd.DataFrame(
        {
            "missing_values_in_row": row_missing,
            "missing_pct_in_row": (row_missing / df.shape[1] * 100).round(2),
        }
    ).sort_values(
        by="missing_values_in_row",
        ascending=False,
    )