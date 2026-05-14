import numpy as np
import pandas as pd


def get_iqr_outlier_summary(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> pd.DataFrame:
    """
    Detect possible numeric outliers using the IQR rule.

    This is not a definitive anomaly detector. It is a quick EDA heuristic.
    """
    rows = []

    for column in numeric_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            lower_bound = q1
            upper_bound = q3
            outlier_count = 0
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_count = int(
                ((series < lower_bound) | (series > upper_bound)).sum()
            )

        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "possible_outliers": outlier_count,
                "possible_outlier_pct": round((outlier_count / len(series)) * 100, 2),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "q1",
                "q3",
                "iqr",
                "lower_bound",
                "upper_bound",
                "possible_outliers",
                "possible_outlier_pct",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by="possible_outliers",
        ascending=False,
    )


def build_numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    if not numeric_columns:
        return pd.DataFrame()

    return df[numeric_columns].describe().T