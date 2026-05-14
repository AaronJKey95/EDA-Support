from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from eda.statistical_tests import (
    calculate_cohens_d,
    calculate_cramers_v_from_chi_square,
    calculate_rank_biserial_correlation,
    prepare_categorical_categorical_data,
    prepare_categorical_numeric_data,
    safe_float,
)


def add_adjusted_p_values(
    df: pd.DataFrame,
    p_column: str = "p_value",
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """
    Add multiple-testing-adjusted p-values to a result table.

    NaN p-values are ignored during adjustment and remain NaN afterwards.
    """
    output = df.copy()

    if output.empty or p_column not in output.columns:
        output["p_value_adjusted"] = np.nan
        output["reject_adjusted"] = False
        return output

    p_values = pd.to_numeric(output[p_column], errors="coerce")
    valid_mask = p_values.notna()

    output["p_value_adjusted"] = np.nan
    output["reject_adjusted"] = False

    if valid_mask.sum() == 0:
        return output

    reject, adjusted_p_values, _, _ = multipletests(
        pvals=p_values.loc[valid_mask].to_numpy(),
        alpha=alpha,
        method=method,
    )

    output.loc[valid_mask, "p_value_adjusted"] = adjusted_p_values
    output.loc[valid_mask, "reject_adjusted"] = reject

    return output


def get_candidate_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
    min_complete_rows: int,
) -> list[str]:
    """Return numeric columns suitable for scanning."""
    candidates = []

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()

        if len(series) < min_complete_rows:
            continue

        if series.nunique(dropna=True) < 2:
            continue

        candidates.append(column)

    return candidates


def get_candidate_categorical_columns(
    df: pd.DataFrame,
    categorical_columns: list[str],
    min_complete_rows: int,
    max_categories: int,
    max_unique_pct: float,
) -> list[str]:
    """
    Return categorical columns suitable for scanning.

    This deliberately skips likely ID/free-text columns.
    """
    candidates = []

    for column in categorical_columns:
        series = df[column].dropna().astype("string")

        if len(series) < min_complete_rows:
            continue

        unique_count = series.nunique(dropna=True)

        if unique_count < 2:
            continue

        unique_pct = unique_count / len(series)

        if unique_count > max_categories:
            continue

        if unique_pct > max_unique_pct:
            continue

        candidates.append(column)

    return candidates


def get_top_categories(
    series: pd.Series,
    min_group_size: int,
    max_levels: int,
) -> list[str]:
    """Return the most common valid categories."""
    counts = series.dropna().astype("string").value_counts()

    return counts[counts >= min_group_size].head(max_levels).index.tolist()


def get_direction_from_difference(difference: float) -> str:
    """Return a simple direction label."""
    if pd.isna(difference):
        return "unknown"

    if difference > 0:
        return "higher"

    if difference < 0:
        return "lower"

    return "no difference"


def scan_numeric_numeric_relationships(
    df: pd.DataFrame,
    numeric_columns: list[str],
    min_complete_rows: int = 20,
    alpha: float = 0.05,
    p_adjust_method: str = "fdr_bh",
    required_column: str | None = None,
) -> pd.DataFrame:
    """
    Scan numeric-vs-numeric relationships.

    If required_column is provided, only pairs containing that column are scanned.
    """
    rows = []

    for x_column, y_column in combinations(numeric_columns, 2):
        if required_column is not None and required_column not in {x_column, y_column}:
            continue

        pair_df = df[[x_column, y_column]].copy()
        pair_df[x_column] = pd.to_numeric(pair_df[x_column], errors="coerce")
        pair_df[y_column] = pd.to_numeric(pair_df[y_column], errors="coerce")
        pair_df = pair_df.dropna()

        n = len(pair_df)

        if n < min_complete_rows:
            continue

        if pair_df[x_column].nunique() < 2 or pair_df[y_column].nunique() < 2:
            continue

        for method_name, test_function in [
            ("Pearson correlation", stats.pearsonr),
            ("Spearman correlation", stats.spearmanr),
        ]:
            try:
                result = test_function(pair_df[x_column], pair_df[y_column])
                statistic = safe_float(result.statistic)
                p_value = safe_float(result.pvalue)
            except Exception:
                statistic = np.nan
                p_value = np.nan

            rows.append(
                {
                    "relationship_type": "numeric_numeric",
                    "x_column": x_column,
                    "y_column": y_column,
                    "x_category": None,
                    "y_category": None,
                    "test_name": method_name,
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_size_name": method_name.replace(" correlation", " r"),
                    "effect_size": statistic,
                    "absolute_effect_size": abs(statistic)
                    if not pd.isna(statistic)
                    else np.nan,
                    "difference": np.nan,
                    "absolute_difference": np.nan,
                    "direction": "positive"
                    if not pd.isna(statistic) and statistic > 0
                    else "negative"
                    if not pd.isna(statistic) and statistic < 0
                    else "none",
                    "n": n,
                    "n_x_category": np.nan,
                    "n_rest": np.nan,
                    "warning": None,
                    "rank_score": abs(statistic) if not pd.isna(statistic) else np.nan,
                }
            )

    results = pd.DataFrame(rows)

    return add_adjusted_p_values(
        results,
        p_column="p_value",
        alpha=alpha,
        method=p_adjust_method,
    )


def scan_category_numeric_overall_relationships(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    min_complete_rows: int = 20,
    min_group_size: int = 20,
    alpha: float = 0.05,
    p_adjust_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Scan overall category-vs-numeric relationships."""
    rows = []

    for categorical_column in categorical_columns:
        for numeric_column in numeric_columns:
            working_df = prepare_categorical_numeric_data(
                df=df,
                categorical_column=categorical_column,
                numeric_column=numeric_column,
            )

            if len(working_df) < min_complete_rows:
                continue

            grouped_values = [
                group[numeric_column].dropna()
                for _, group in working_df.groupby(categorical_column)
            ]

            valid_groups = [
                group for group in grouped_values if len(group) >= min_group_size
            ]

            if len(valid_groups) < 2:
                continue

            n = int(sum(len(group) for group in valid_groups))
            k = len(valid_groups)

            try:
                kruskal_result = stats.kruskal(*valid_groups)
                kruskal_statistic = safe_float(kruskal_result.statistic)
                kruskal_p_value = safe_float(kruskal_result.pvalue)
            except Exception:
                kruskal_statistic = np.nan
                kruskal_p_value = np.nan

            if not pd.isna(kruskal_statistic) and n > k:
                epsilon_squared = max(
                    0,
                    (kruskal_statistic - k + 1) / (n - k),
                )
                epsilon_squared = safe_float(epsilon_squared)
            else:
                epsilon_squared = np.nan

            try:
                anova_result = stats.f_oneway(*valid_groups)
                anova_statistic = safe_float(anova_result.statistic)
                anova_p_value = safe_float(anova_result.pvalue)
            except Exception:
                anova_statistic = np.nan
                anova_p_value = np.nan

            rows.append(
                {
                    "relationship_type": "category_numeric_overall",
                    "x_column": categorical_column,
                    "y_column": numeric_column,
                    "x_category": None,
                    "y_category": None,
                    "test_name": "Kruskal-Wallis",
                    "statistic": kruskal_statistic,
                    "p_value": kruskal_p_value,
                    "effect_size_name": "epsilon-squared",
                    "effect_size": epsilon_squared,
                    "absolute_effect_size": abs(epsilon_squared)
                    if not pd.isna(epsilon_squared)
                    else np.nan,
                    "difference": np.nan,
                    "absolute_difference": np.nan,
                    "direction": "overall difference",
                    "n": n,
                    "n_x_category": np.nan,
                    "n_rest": np.nan,
                    "number_of_groups": k,
                    "secondary_test_name": "One-way ANOVA",
                    "secondary_statistic": anova_statistic,
                    "secondary_p_value": anova_p_value,
                    "warning": None,
                    "rank_score": epsilon_squared,
                }
            )

    results = pd.DataFrame(rows)

    return add_adjusted_p_values(
        results,
        p_column="p_value",
        alpha=alpha,
        method=p_adjust_method,
    )


def scan_category_numeric_contrasts(
    df: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    min_complete_rows: int = 20,
    min_group_size: int = 20,
    max_levels_per_column: int = 10,
    alpha: float = 0.05,
    p_adjust_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Scan category-vs-rest contrasts for categorical-vs-numeric relationships."""
    rows = []

    for categorical_column in categorical_columns:
        for numeric_column in numeric_columns:
            working_df = prepare_categorical_numeric_data(
                df=df,
                categorical_column=categorical_column,
                numeric_column=numeric_column,
            )

            if len(working_df) < min_complete_rows:
                continue

            categories = get_top_categories(
                series=working_df[categorical_column],
                min_group_size=min_group_size,
                max_levels=max_levels_per_column,
            )

            for category in categories:
                selected_mask = working_df[categorical_column] == str(category)

                selected_values = working_df.loc[
                    selected_mask,
                    numeric_column,
                ].dropna()

                rest_values = working_df.loc[
                    ~selected_mask,
                    numeric_column,
                ].dropna()

                if len(selected_values) < min_group_size:
                    continue

                if len(rest_values) < min_group_size:
                    continue

                cohens_d = calculate_cohens_d(selected_values, rest_values)

                try:
                    mann_whitney_result = stats.mannwhitneyu(
                        selected_values,
                        rest_values,
                        alternative="two-sided",
                    )
                    mw_statistic = safe_float(mann_whitney_result.statistic)
                    mw_p_value = safe_float(mann_whitney_result.pvalue)
                except Exception:
                    mw_statistic = np.nan
                    mw_p_value = np.nan

                rank_biserial = calculate_rank_biserial_correlation(
                    mann_whitney_u=mw_statistic,
                    n_group_a=len(selected_values),
                    n_group_b=len(rest_values),
                )

                try:
                    welch_result = stats.ttest_ind(
                        selected_values,
                        rest_values,
                        equal_var=False,
                    )
                    welch_statistic = safe_float(welch_result.statistic)
                    welch_p_value = safe_float(welch_result.pvalue)
                except Exception:
                    welch_statistic = np.nan
                    welch_p_value = np.nan

                mean_difference = safe_float(
                    selected_values.mean() - rest_values.mean()
                )

                median_difference = safe_float(
                    selected_values.median() - rest_values.median()
                )

                rows.append(
                    {
                        "relationship_type": "category_numeric_contrast",
                        "x_column": categorical_column,
                        "y_column": numeric_column,
                        "x_category": category,
                        "y_category": None,
                        "test_name": "Mann-Whitney U",
                        "statistic": mw_statistic,
                        "p_value": mw_p_value,
                        "effect_size_name": "rank-biserial correlation",
                        "effect_size": rank_biserial,
                        "absolute_effect_size": abs(rank_biserial)
                        if not pd.isna(rank_biserial)
                        else np.nan,
                        "difference": median_difference,
                        "absolute_difference": abs(median_difference)
                        if not pd.isna(median_difference)
                        else np.nan,
                        "mean_difference": mean_difference,
                        "median_difference": median_difference,
                        "direction": get_direction_from_difference(
                            median_difference
                        ),
                        "n": len(selected_values) + len(rest_values),
                        "n_x_category": len(selected_values),
                        "n_rest": len(rest_values),
                        "secondary_test_name": "Welch t-test",
                        "secondary_statistic": welch_statistic,
                        "secondary_p_value": welch_p_value,
                        "cohens_d": cohens_d,
                        "warning": None,
                        "rank_score": abs(rank_biserial)
                        if not pd.isna(rank_biserial)
                        else np.nan,
                    }
                )

    results = pd.DataFrame(rows)

    return add_adjusted_p_values(
        results,
        p_column="p_value",
        alpha=alpha,
        method=p_adjust_method,
    )


def scan_category_category_overall_relationships(
    df: pd.DataFrame,
    categorical_columns: list[str],
    min_complete_rows: int = 20,
    min_group_size: int = 20,
    alpha: float = 0.05,
    p_adjust_method: str = "fdr_bh",
    required_column: str | None = None,
) -> pd.DataFrame:
    """
    Scan overall categorical-vs-categorical associations.

    If required_column is provided, only pairs containing that column are scanned.
    """
    rows = []

    for x_column, y_column in combinations(categorical_columns, 2):
        if required_column is not None and required_column not in {x_column, y_column}:
            continue

        working_df = prepare_categorical_categorical_data(
            df=df,
            x_column=x_column,
            y_column=y_column,
        )

        if len(working_df) < min_complete_rows:
            continue

        x_valid_categories = get_top_categories(
            working_df[x_column],
            min_group_size=min_group_size,
            max_levels=10_000,
        )

        y_valid_categories = get_top_categories(
            working_df[y_column],
            min_group_size=min_group_size,
            max_levels=10_000,
        )

        working_df = working_df[
            working_df[x_column].isin(x_valid_categories)
            & working_df[y_column].isin(y_valid_categories)
        ]

        if len(working_df) < min_complete_rows:
            continue

        observed = pd.crosstab(
            working_df[x_column],
            working_df[y_column],
        )

        if observed.shape[0] < 2 or observed.shape[1] < 2:
            continue

        try:
            chi2_result = stats.chi2_contingency(
                observed,
                correction=False,
            )

            chi2_statistic = safe_float(chi2_result.statistic)
            chi2_p_value = safe_float(chi2_result.pvalue)
            dof = int(chi2_result.dof)
            expected = chi2_result.expected_freq
        except Exception:
            chi2_statistic = np.nan
            chi2_p_value = np.nan
            dof = np.nan
            expected = np.full(observed.shape, np.nan)

        n = int(observed.to_numpy().sum())
        n_rows, n_columns = observed.shape

        cramers_v = calculate_cramers_v_from_chi_square(
            chi_square_statistic=chi2_statistic,
            n_observations=n,
            n_rows=n_rows,
            n_columns=n_columns,
        )

        expected_values = np.asarray(expected)

        if np.isnan(expected_values).all():
            minimum_expected_count = np.nan
            cells_expected_below_5 = np.nan
            pct_cells_expected_below_5 = np.nan
            warning = "Expected counts could not be calculated."
        else:
            minimum_expected_count = safe_float(np.nanmin(expected_values))
            cells_expected_below_5 = int((expected_values < 5).sum())
            total_cells = int(expected_values.size)
            pct_cells_expected_below_5 = safe_float(
                cells_expected_below_5 / total_cells * 100
            )

            warning = (
                "Sparse expected counts."
                if cells_expected_below_5 > 0
                else None
            )

        rows.append(
            {
                "relationship_type": "category_category_overall",
                "x_column": x_column,
                "y_column": y_column,
                "x_category": None,
                "y_category": None,
                "test_name": "Chi-square test of independence",
                "statistic": chi2_statistic,
                "p_value": chi2_p_value,
                "effect_size_name": "Cramér's V",
                "effect_size": cramers_v,
                "absolute_effect_size": abs(cramers_v)
                if not pd.isna(cramers_v)
                else np.nan,
                "difference": np.nan,
                "absolute_difference": np.nan,
                "direction": "overall association",
                "n": n,
                "n_x_category": np.nan,
                "n_rest": np.nan,
                "degrees_of_freedom": dof,
                "minimum_expected_count": minimum_expected_count,
                "cells_expected_below_5": cells_expected_below_5,
                "pct_cells_expected_below_5": pct_cells_expected_below_5,
                "warning": warning,
                "rank_score": cramers_v,
            }
        )

    results = pd.DataFrame(rows)

    return add_adjusted_p_values(
        results,
        p_column="p_value",
        alpha=alpha,
        method=p_adjust_method,
    )


def scan_category_category_contrasts(
    df: pd.DataFrame,
    categorical_columns: list[str],
    min_complete_rows: int = 20,
    min_group_size: int = 20,
    max_levels_per_column: int = 8,
    alpha: float = 0.05,
    p_adjust_method: str = "fdr_bh",
    required_column: str | None = None,
) -> pd.DataFrame:
    """
    Scan selected category-pair contrasts.

    For each pair:
    - x category vs rest
    - y category vs rest

    If required_column is provided, only pairs containing that column are scanned.
    """
    rows = []

    for x_column, y_column in combinations(categorical_columns, 2):
        if required_column is not None and required_column not in {x_column, y_column}:
            continue

        working_df = prepare_categorical_categorical_data(
            df=df,
            x_column=x_column,
            y_column=y_column,
        )

        if len(working_df) < min_complete_rows:
            continue

        x_categories = get_top_categories(
            working_df[x_column],
            min_group_size=min_group_size,
            max_levels=max_levels_per_column,
        )

        y_categories = get_top_categories(
            working_df[y_column],
            min_group_size=min_group_size,
            max_levels=max_levels_per_column,
        )

        for x_category in x_categories:
            for y_category in y_categories:
                x_selected = working_df[x_column] == str(x_category)
                y_selected = working_df[y_column] == str(y_category)

                selected_x_total = int(x_selected.sum())
                rest_x_total = int((~x_selected).sum())

                if selected_x_total < min_group_size:
                    continue

                if rest_x_total < min_group_size:
                    continue

                a = int((x_selected & y_selected).sum())
                b = int((x_selected & ~y_selected).sum())
                c = int((~x_selected & y_selected).sum())
                d = int((~x_selected & ~y_selected).sum())

                contrast_table = np.array([[a, b], [c, d]])

                selected_y_rate_in_selected_x = (
                    a / selected_x_total if selected_x_total > 0 else np.nan
                )

                selected_y_rate_in_rest_x = (
                    c / rest_x_total if rest_x_total > 0 else np.nan
                )

                rate_difference = (
                    selected_y_rate_in_selected_x - selected_y_rate_in_rest_x
                    if not pd.isna(selected_y_rate_in_selected_x)
                    and not pd.isna(selected_y_rate_in_rest_x)
                    else np.nan
                )

                try:
                    fisher_result = stats.fisher_exact(
                        contrast_table,
                        alternative="two-sided",
                    )
                    odds_ratio = safe_float(fisher_result.statistic)
                    fisher_p_value = safe_float(fisher_result.pvalue)
                except Exception:
                    odds_ratio = np.nan
                    fisher_p_value = np.nan

                try:
                    chi2_result = stats.chi2_contingency(
                        contrast_table,
                        correction=False,
                    )
                    chi2_statistic = safe_float(chi2_result.statistic)
                    chi2_p_value = safe_float(chi2_result.pvalue)
                except Exception:
                    chi2_statistic = np.nan
                    chi2_p_value = np.nan

                rows.append(
                    {
                        "relationship_type": "category_category_contrast",
                        "x_column": x_column,
                        "y_column": y_column,
                        "x_category": x_category,
                        "y_category": y_category,
                        "test_name": "Fisher's exact test",
                        "statistic": odds_ratio,
                        "p_value": fisher_p_value,
                        "effect_size_name": "odds ratio",
                        "effect_size": odds_ratio,
                        "absolute_effect_size": abs(np.log(odds_ratio))
                        if not pd.isna(odds_ratio)
                        and odds_ratio > 0
                        and np.isfinite(odds_ratio)
                        else np.nan,
                        "difference": safe_float(rate_difference),
                        "absolute_difference": abs(rate_difference)
                        if not pd.isna(rate_difference)
                        else np.nan,
                        "direction": "enriched"
                        if not pd.isna(rate_difference) and rate_difference > 0
                        else "depleted"
                        if not pd.isna(rate_difference) and rate_difference < 0
                        else "no difference",
                        "n": selected_x_total + rest_x_total,
                        "n_x_category": selected_x_total,
                        "n_rest": rest_x_total,
                        "selected_y_count_in_selected_x": a,
                        "selected_y_count_in_rest": c,
                        "selected_y_rate_in_selected_x": safe_float(
                            selected_y_rate_in_selected_x
                        ),
                        "selected_y_rate_in_rest": safe_float(
                            selected_y_rate_in_rest_x
                        ),
                        "secondary_test_name": "Chi-square test on 2x2 contrast",
                        "secondary_statistic": chi2_statistic,
                        "secondary_p_value": chi2_p_value,
                        "warning": None,
                        "rank_score": abs(rate_difference)
                        if not pd.isna(rate_difference)
                        else np.nan,
                    }
                )

    results = pd.DataFrame(rows)

    return add_adjusted_p_values(
        results,
        p_column="p_value",
        alpha=alpha,
        method=p_adjust_method,
    )


def build_interesting_relationships_report(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    config: dict,
) -> dict:
    """
    Build the interesting relationships report.

    If target_column is supplied in config, only relationships involving that
    target are scanned.
    """
    min_complete_rows = int(config.get("min_complete_rows", 20))
    min_group_size = int(config.get("min_group_size", 20))
    max_categories = int(config.get("max_categories", 30))
    max_unique_pct = float(config.get("max_unique_pct", 0.95))
    max_levels_per_column = int(config.get("max_levels_per_column", 10))
    alpha = float(config.get("alpha", 0.05))
    p_adjust_method = config.get("p_adjust_method", "fdr_bh")
    include_category_category_contrasts = bool(
        config.get("include_category_category_contrasts", True)
    )
    target_column = config.get("target_column")

    candidate_numeric_columns = get_candidate_numeric_columns(
        df=df,
        numeric_columns=numeric_columns,
        min_complete_rows=min_complete_rows,
    )

    candidate_categorical_columns = get_candidate_categorical_columns(
        df=df,
        categorical_columns=categorical_columns,
        min_complete_rows=min_complete_rows,
        max_categories=max_categories,
        max_unique_pct=max_unique_pct,
    )

    empty = pd.DataFrame()

    target_type = None

    if target_column is not None:
        if target_column in candidate_numeric_columns:
            target_type = "numeric"

        elif target_column in candidate_categorical_columns:
            target_type = "categorical"

        else:
            return {
                "target_column": target_column,
                "target_type": "unsupported_or_filtered_out",
                "warning": (
                    f"`{target_column}` was not scanned because it was not eligible "
                    "under the current scan controls. Try lowering minimum rows/group size, "
                    "increasing max categories, or checking the column type."
                ),
                "candidate_numeric_columns": candidate_numeric_columns,
                "candidate_categorical_columns": candidate_categorical_columns,
                "numeric_numeric": empty,
                "category_numeric_overall": empty,
                "category_numeric_contrasts": empty,
                "category_category_overall": empty,
                "category_category_contrasts": empty,
            }

    if target_column is None:
        numeric_numeric = scan_numeric_numeric_relationships(
            df=df,
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_numeric_overall = scan_category_numeric_overall_relationships(
            df=df,
            categorical_columns=candidate_categorical_columns,
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_numeric_contrasts = scan_category_numeric_contrasts(
            df=df,
            categorical_columns=candidate_categorical_columns,
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            max_levels_per_column=max_levels_per_column,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_category_overall = scan_category_category_overall_relationships(
            df=df,
            categorical_columns=candidate_categorical_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        if include_category_category_contrasts:
            category_category_contrasts = scan_category_category_contrasts(
                df=df,
                categorical_columns=candidate_categorical_columns,
                min_complete_rows=min_complete_rows,
                min_group_size=min_group_size,
                max_levels_per_column=max_levels_per_column,
                alpha=alpha,
                p_adjust_method=p_adjust_method,
            )
        else:
            category_category_contrasts = empty

    elif target_type == "numeric":
        numeric_numeric = scan_numeric_numeric_relationships(
            df=df,
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
            required_column=target_column,
        )

        category_numeric_overall = scan_category_numeric_overall_relationships(
            df=df,
            categorical_columns=candidate_categorical_columns,
            numeric_columns=[target_column],
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_numeric_contrasts = scan_category_numeric_contrasts(
            df=df,
            categorical_columns=candidate_categorical_columns,
            numeric_columns=[target_column],
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            max_levels_per_column=max_levels_per_column,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_category_overall = empty
        category_category_contrasts = empty

    elif target_type == "categorical":
        numeric_numeric = empty

        category_numeric_overall = scan_category_numeric_overall_relationships(
            df=df,
            categorical_columns=[target_column],
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        category_numeric_contrasts = scan_category_numeric_contrasts(
            df=df,
            categorical_columns=[target_column],
            numeric_columns=candidate_numeric_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            max_levels_per_column=max_levels_per_column,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
        )

        target_ordered_categorical_columns = [
            column
            for column in candidate_categorical_columns
            if column != target_column
        ] + [target_column]

        category_category_overall = scan_category_category_overall_relationships(
            df=df,
            categorical_columns=target_ordered_categorical_columns,
            min_complete_rows=min_complete_rows,
            min_group_size=min_group_size,
            alpha=alpha,
            p_adjust_method=p_adjust_method,
            required_column=target_column,
        )

        if include_category_category_contrasts:
            category_category_contrasts = scan_category_category_contrasts(
                df=df,
                categorical_columns=target_ordered_categorical_columns,
                min_complete_rows=min_complete_rows,
                min_group_size=min_group_size,
                max_levels_per_column=max_levels_per_column,
                alpha=alpha,
                p_adjust_method=p_adjust_method,
                required_column=target_column,
            )
        else:
            category_category_contrasts = empty

    return {
        "target_column": target_column,
        "target_type": target_type,
        "candidate_numeric_columns": candidate_numeric_columns,
        "candidate_categorical_columns": candidate_categorical_columns,
        "numeric_numeric": numeric_numeric,
        "category_numeric_overall": category_numeric_overall,
        "category_numeric_contrasts": category_numeric_contrasts,
        "category_category_overall": category_category_overall,
        "category_category_contrasts": category_category_contrasts,
    }