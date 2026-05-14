import pandas as pd


def get_basic_metadata(df: pd.DataFrame) -> dict:
    """Return high-level dataset metadata."""
    return {
        "Rows": df.shape[0],
        "Columns": df.shape[1],
        "Duplicate rows": int(df.duplicated().sum()),
        "Total missing values": int(df.isna().sum().sum()),
        "Memory usage MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level profile table."""
    unique_counts = df.nunique(dropna=True)

    if len(df) == 0:
        unique_pct = [0 for _ in df.columns]
    else:
        unique_pct = ((unique_counts / len(df)) * 100).round(2).values

    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "non_null_count": df.notna().sum().values,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(2),
            "unique_count": unique_counts.values,
            "unique_pct": unique_pct,
        }
    )

    return profile.sort_values(
        by=["missing_pct", "unique_count"],
        ascending=[False, False],
    )