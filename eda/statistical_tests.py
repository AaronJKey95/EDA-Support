import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower


def safe_float(value) -> float:
    """Convert numpy/scipy values to plain Python floats for cleaner display."""
    if pd.isna(value):
        return np.nan

    return float(value)


def format_p_value(p_value: float) -> str:
    """Format p-values for user-facing display."""
    if pd.isna(p_value):
        return "N/A"

    if p_value < 0.001:
        return "< 0.001"

    return f"{p_value:.4f}"


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """Return a simple p-value interpretation."""
    if pd.isna(p_value):
        return "Unable to calculate."

    if p_value < alpha:
        return f"Statistically significant at alpha = {alpha}."

    return f"Not statistically significant at alpha = {alpha}."


def prepare_categorical_numeric_data(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
) -> pd.DataFrame:
    """
    Prepare a two-column dataframe for categorical-vs-numeric testing.

    Missing values are removed because the statistical tests need complete
    category/outcome pairs.
    """
    working_df = df[[categorical_column, numeric_column]].copy()

    working_df[numeric_column] = pd.to_numeric(
        working_df[numeric_column],
        errors="coerce",
    )

    working_df = working_df.dropna(
        subset=[categorical_column, numeric_column],
    )

    working_df[categorical_column] = working_df[categorical_column].astype("string")

    return working_df


def build_group_summary(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
) -> pd.DataFrame:
    """Build descriptive statistics for each category."""
    working_df = prepare_categorical_numeric_data(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
    )

    if working_df.empty:
        return pd.DataFrame()

    summary = (
        working_df.groupby(categorical_column)[numeric_column]
        .agg(
            n="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
        .sort_values(by="n", ascending=False)
    )

    return summary


def calculate_cohens_d(
    group_a: pd.Series,
    group_b: pd.Series,
) -> float:
    """
    Calculate Cohen's d for two independent groups.

    Positive value means group_a has a higher mean than group_b.
    """
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        return np.nan

    var_a = group_a.var(ddof=1)
    var_b = group_b.var(ddof=1)

    pooled_variance = (
        ((n_a - 1) * var_a) + ((n_b - 1) * var_b)
    ) / (n_a + n_b - 2)

    pooled_sd = np.sqrt(pooled_variance)

    if pooled_sd == 0:
        return np.nan

    return safe_float((group_a.mean() - group_b.mean()) / pooled_sd)


def calculate_rank_biserial_correlation(
    mann_whitney_u: float,
    n_group_a: int,
    n_group_b: int,
) -> float:
    """
    Calculate rank-biserial correlation from Mann-Whitney U.

    Positive value means group_a tends to have larger values than group_b.
    """
    if pd.isna(mann_whitney_u) or n_group_a == 0 or n_group_b == 0:
        return np.nan

    return safe_float((2 * mann_whitney_u) / (n_group_a * n_group_b) - 1)


def calculate_approximate_ttest_power(
    effect_size: float,
    n_group: int,
    n_rest: int,
    alpha: float = 0.05,
) -> float:
    """
    Approximate observed power for a two-sample t-test.

    This uses observed Cohen's d, so it is useful as an EDA signal, not as
    formal study design evidence.
    """
    if pd.isna(effect_size) or n_group < 2 or n_rest < 2:
        return np.nan

    ratio = n_rest / n_group

    try:
        power = TTestIndPower().power(
            effect_size=abs(effect_size),
            nobs1=n_group,
            ratio=ratio,
            alpha=alpha,
            alternative="two-sided",
        )
        return safe_float(power)
    except Exception:
        return np.nan


def run_overall_group_tests(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    alpha: float = 0.05,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Run overall tests comparing the numeric outcome across all categories.

    Includes:
    - One-way ANOVA
    - Kruskal-Wallis
    """
    working_df = prepare_categorical_numeric_data(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
    )

    grouped_values = [
        group[numeric_column].dropna()
        for _, group in working_df.groupby(categorical_column)
    ]

    valid_groups = [
        group
        for group in grouped_values
        if len(group) >= min_group_size
    ]

    rows = []

    if len(valid_groups) < 2:
        return pd.DataFrame(
            [
                {
                    "test": "One-way ANOVA",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "p_value_display": "N/A",
                    "interpretation": "At least two groups with sufficient observations are required.",
                },
                {
                    "test": "Kruskal-Wallis",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "p_value_display": "N/A",
                    "interpretation": "At least two groups with sufficient observations are required.",
                },
            ]
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            anova_result = stats.f_oneway(*valid_groups)

        anova_statistic = safe_float(anova_result.statistic)
        anova_p_value = safe_float(anova_result.pvalue)
    except Exception:
        anova_statistic = np.nan
        anova_p_value = np.nan

    rows.append(
        {
            "test": "One-way ANOVA",
            "statistic": anova_statistic,
            "p_value": anova_p_value,
            "p_value_display": format_p_value(anova_p_value),
            "interpretation": interpret_p_value(anova_p_value, alpha),
        }
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kruskal_result = stats.kruskal(*valid_groups)

        kruskal_statistic = safe_float(kruskal_result.statistic)
        kruskal_p_value = safe_float(kruskal_result.pvalue)
    except Exception:
        kruskal_statistic = np.nan
        kruskal_p_value = np.nan

    rows.append(
        {
            "test": "Kruskal-Wallis",
            "statistic": kruskal_statistic,
            "p_value": kruskal_p_value,
            "p_value_display": format_p_value(kruskal_p_value),
            "interpretation": interpret_p_value(kruskal_p_value, alpha),
        }
    )

    return pd.DataFrame(rows)


def run_selected_category_vs_rest_test(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    selected_category: str,
    alpha: float = 0.05,
) -> dict:
    """
    Compare one selected category against the remainder of the cohort.

    Includes:
    - mean difference
    - median difference
    - Cohen's d
    - Welch t-test
    - Mann-Whitney U
    - rank-biserial correlation
    - approximate observed power
    """
    working_df = prepare_categorical_numeric_data(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
    )

    if working_df.empty:
        return {
            "error": "No complete rows available for this comparison.",
        }

    selected_category = str(selected_category)

    selected_mask = working_df[categorical_column] == selected_category

    selected_values = working_df.loc[selected_mask, numeric_column].dropna()
    rest_values = working_df.loc[~selected_mask, numeric_column].dropna()

    if len(selected_values) < 2 or len(rest_values) < 2:
        return {
            "error": "The selected category or remainder group has fewer than 2 observations.",
        }

    cohens_d = calculate_cohens_d(selected_values, rest_values)

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

    try:
        mann_whitney_result = stats.mannwhitneyu(
            selected_values,
            rest_values,
            alternative="two-sided",
        )

        mann_whitney_statistic = safe_float(mann_whitney_result.statistic)
        mann_whitney_p_value = safe_float(mann_whitney_result.pvalue)
    except Exception:
        mann_whitney_statistic = np.nan
        mann_whitney_p_value = np.nan

    rank_biserial = calculate_rank_biserial_correlation(
        mann_whitney_u=mann_whitney_statistic,
        n_group_a=len(selected_values),
        n_group_b=len(rest_values),
    )

    approximate_power = calculate_approximate_ttest_power(
        effect_size=cohens_d,
        n_group=len(selected_values),
        n_rest=len(rest_values),
        alpha=alpha,
    )

    contrast_summary = {
        "selected_category": selected_category,
        "selected_n": len(selected_values),
        "rest_n": len(rest_values),
        "selected_mean": safe_float(selected_values.mean()),
        "rest_mean": safe_float(rest_values.mean()),
        "mean_difference": safe_float(selected_values.mean() - rest_values.mean()),
        "selected_median": safe_float(selected_values.median()),
        "rest_median": safe_float(rest_values.median()),
        "median_difference": safe_float(selected_values.median() - rest_values.median()),
        "cohens_d": cohens_d,
        "rank_biserial_correlation": rank_biserial,
        "approximate_ttest_power": approximate_power,
    }

    test_results = pd.DataFrame(
        [
            {
                "test": "Welch t-test",
                "statistic": welch_statistic,
                "p_value": welch_p_value,
                "p_value_display": format_p_value(welch_p_value),
                "effect_size_name": "Cohen's d",
                "effect_size": cohens_d,
                "interpretation": interpret_p_value(welch_p_value, alpha),
            },
            {
                "test": "Mann-Whitney U",
                "statistic": mann_whitney_statistic,
                "p_value": mann_whitney_p_value,
                "p_value_display": format_p_value(mann_whitney_p_value),
                "effect_size_name": "Rank-biserial correlation",
                "effect_size": rank_biserial,
                "interpretation": interpret_p_value(mann_whitney_p_value, alpha),
            },
        ]
    )

    return {
        "contrast_summary": contrast_summary,
        "test_results": test_results,
    }


def run_categorical_numeric_test_report(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
    selected_category: str,
    alpha: float = 0.05,
) -> dict:
    """
    Build a full categorical-vs-numeric statistical test report.
    """
    group_summary = build_group_summary(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
    )

    overall_tests = run_overall_group_tests(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
        alpha=alpha,
    )

    selected_vs_rest = run_selected_category_vs_rest_test(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
        selected_category=selected_category,
        alpha=alpha,
    )

    return {
        "group_summary": group_summary,
        "overall_tests": overall_tests,
        "selected_vs_rest": selected_vs_rest,
    }

def prepare_categorical_categorical_data(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> pd.DataFrame:
    """
    Prepare two categorical columns for categorical-vs-categorical testing.

    Missing values are removed because contingency-table tests require complete
    observed category pairs.
    """
    working_df = df[[x_column, y_column]].copy()

    working_df = working_df.dropna(
        subset=[x_column, y_column],
    )

    working_df[x_column] = working_df[x_column].astype("string")
    working_df[y_column] = working_df[y_column].astype("string")

    return working_df


def calculate_cramers_v_from_chi_square(
    chi_square_statistic: float,
    n_observations: int,
    n_rows: int,
    n_columns: int,
) -> float:
    """
    Calculate Cramér's V from a chi-square statistic.

    Cramér's V ranges from 0 to 1.
    0 means no association.
    1 means maximum association.
    """
    if (
        pd.isna(chi_square_statistic)
        or n_observations <= 0
        or n_rows < 2
        or n_columns < 2
    ):
        return np.nan

    denominator = min(n_rows - 1, n_columns - 1)

    if denominator <= 0:
        return np.nan

    cramers_v = np.sqrt(
        (chi_square_statistic / n_observations) / denominator
    )

    return safe_float(cramers_v)


def calculate_tschuprows_t_from_chi_square(
    chi_square_statistic: float,
    n_observations: int,
    n_rows: int,
    n_columns: int,
) -> float:
    """
    Calculate Tschuprow's T from a chi-square statistic.

    This is another association measure for contingency tables.
    """
    if (
        pd.isna(chi_square_statistic)
        or n_observations <= 0
        or n_rows < 2
        or n_columns < 2
    ):
        return np.nan

    denominator = np.sqrt((n_rows - 1) * (n_columns - 1))

    if denominator <= 0:
        return np.nan

    tschuprows_t = np.sqrt(
        (chi_square_statistic / n_observations) / denominator
    )

    return safe_float(tschuprows_t)


def build_categorical_crosstab_report(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> dict:
    """
    Build observed counts, percentages, expected counts, and residuals
    for two categorical columns.
    """
    working_df = prepare_categorical_categorical_data(
        df=df,
        x_column=x_column,
        y_column=y_column,
    )

    if working_df.empty:
        return {
            "error": "No complete rows are available for categorical relationship testing.",
        }

    observed = pd.crosstab(
        working_df[x_column],
        working_df[y_column],
    )

    if observed.shape[0] < 2 or observed.shape[1] < 2:
        return {
            "error": "At least two categories are required in each column.",
        }

    row_percentages = observed.div(
        observed.sum(axis=1),
        axis=0,
    ) * 100

    column_percentages = observed.div(
        observed.sum(axis=0),
        axis=1,
    ) * 100

    try:
        chi2_result = stats.chi2_contingency(
            observed,
            correction=False,
        )

        chi_square_statistic = safe_float(chi2_result.statistic)
        chi_square_p_value = safe_float(chi2_result.pvalue)
        degrees_of_freedom = int(chi2_result.dof)

        expected = pd.DataFrame(
            chi2_result.expected_freq,
            index=observed.index,
            columns=observed.columns,
        )

        standardized_residuals = (observed - expected) / np.sqrt(expected)

    except Exception:
        chi_square_statistic = np.nan
        chi_square_p_value = np.nan
        degrees_of_freedom = np.nan

        expected = pd.DataFrame(
            np.nan,
            index=observed.index,
            columns=observed.columns,
        )

        standardized_residuals = pd.DataFrame(
            np.nan,
            index=observed.index,
            columns=observed.columns,
        )

    expected_values = expected.to_numpy()

    expected_count_summary = {
        "minimum_expected_count": safe_float(np.nanmin(expected_values)),
        "cells_with_expected_count_below_5": int((expected_values < 5).sum()),
        "total_cells": int(expected_values.size),
        "pct_cells_with_expected_count_below_5": safe_float(
            ((expected_values < 5).sum() / expected_values.size) * 100
        ),
    }

    n_observations = int(observed.to_numpy().sum())
    n_rows, n_columns = observed.shape

    cramers_v = calculate_cramers_v_from_chi_square(
        chi_square_statistic=chi_square_statistic,
        n_observations=n_observations,
        n_rows=n_rows,
        n_columns=n_columns,
    )

    tschuprows_t = calculate_tschuprows_t_from_chi_square(
        chi_square_statistic=chi_square_statistic,
        n_observations=n_observations,
        n_rows=n_rows,
        n_columns=n_columns,
    )

    overall_tests = pd.DataFrame(
        [
            {
                "test": "Chi-square test of independence",
                "statistic": chi_square_statistic,
                "degrees_of_freedom": degrees_of_freedom,
                "p_value": chi_square_p_value,
                "p_value_display": format_p_value(chi_square_p_value),
                "effect_size_name": "Cramér's V",
                "effect_size": cramers_v,
                "interpretation": interpret_p_value(chi_square_p_value),
            }
        ]
    )

    association_summary = pd.DataFrame(
        [
            {
                "measure": "Cramér's V",
                "value": cramers_v,
                "interpretation": "0 means no association; 1 means maximum association.",
            },
            {
                "measure": "Tschuprow's T",
                "value": tschuprows_t,
                "interpretation": "Alternative categorical association measure.",
            },
        ]
    )

    fisher_exact_result = None

    if observed.shape == (2, 2):
        try:
            fisher_result = stats.fisher_exact(
                observed.to_numpy(),
                alternative="two-sided",
            )

            fisher_exact_result = {
                "test": "Fisher's exact test",
                "odds_ratio": safe_float(fisher_result.statistic),
                "p_value": safe_float(fisher_result.pvalue),
                "p_value_display": format_p_value(fisher_result.pvalue),
                "interpretation": interpret_p_value(fisher_result.pvalue),
            }

        except Exception:
            fisher_exact_result = {
                "test": "Fisher's exact test",
                "odds_ratio": np.nan,
                "p_value": np.nan,
                "p_value_display": "N/A",
                "interpretation": "Unable to calculate.",
            }

    return {
        "observed": observed,
        "row_percentages": row_percentages,
        "column_percentages": column_percentages,
        "expected": expected,
        "standardized_residuals": standardized_residuals,
        "expected_count_summary": expected_count_summary,
        "overall_tests": overall_tests,
        "association_summary": association_summary,
        "fisher_exact_result": fisher_exact_result,
    }


def run_selected_category_pair_contrast(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    selected_x_category: str,
    selected_y_category: str,
) -> dict:
    """
    Build and test a 2x2 contrast:

    selected X category vs all other X categories
    selected Y category vs all other Y categories

    Example:
    Instagram vs not Instagram
    High addiction vs not high addiction
    """
    working_df = prepare_categorical_categorical_data(
        df=df,
        x_column=x_column,
        y_column=y_column,
    )

    if working_df.empty:
        return {
            "error": "No complete rows are available for the selected contrast.",
        }

    selected_x_category = str(selected_x_category)
    selected_y_category = str(selected_y_category)

    x_selected_label = f"{x_column} = {selected_x_category}"
    x_rest_label = f"{x_column} != {selected_x_category}"

    y_selected_label = f"{y_column} = {selected_y_category}"
    y_rest_label = f"{y_column} != {selected_y_category}"

    contrast_df = pd.DataFrame(
        {
            "x_group": np.where(
                working_df[x_column] == selected_x_category,
                x_selected_label,
                x_rest_label,
            ),
            "y_group": np.where(
                working_df[y_column] == selected_y_category,
                y_selected_label,
                y_rest_label,
            ),
        }
    )

    contrast_table = pd.crosstab(
        contrast_df["x_group"],
        contrast_df["y_group"],
    )

    contrast_table = contrast_table.reindex(
        index=[x_selected_label, x_rest_label],
        columns=[y_selected_label, y_rest_label],
        fill_value=0,
    )

    selected_x_total = int(contrast_table.loc[x_selected_label].sum())
    rest_x_total = int(contrast_table.loc[x_rest_label].sum())

    selected_y_in_selected_x = int(
        contrast_table.loc[x_selected_label, y_selected_label]
    )

    selected_y_in_rest_x = int(
        contrast_table.loc[x_rest_label, y_selected_label]
    )

    selected_x_rate = (
        selected_y_in_selected_x / selected_x_total
        if selected_x_total > 0
        else np.nan
    )

    rest_x_rate = (
        selected_y_in_rest_x / rest_x_total
        if rest_x_total > 0
        else np.nan
    )

    rate_difference = (
        selected_x_rate - rest_x_rate
        if not pd.isna(selected_x_rate) and not pd.isna(rest_x_rate)
        else np.nan
    )

    try:
        fisher_result = stats.fisher_exact(
            contrast_table.to_numpy(),
            alternative="two-sided",
        )

        fisher_odds_ratio = safe_float(fisher_result.statistic)
        fisher_p_value = safe_float(fisher_result.pvalue)

    except Exception:
        fisher_odds_ratio = np.nan
        fisher_p_value = np.nan

    try:
        chi2_result = stats.chi2_contingency(
            contrast_table,
            correction=False,
        )

        chi_square_statistic = safe_float(chi2_result.statistic)
        chi_square_p_value = safe_float(chi2_result.pvalue)

    except Exception:
        chi_square_statistic = np.nan
        chi_square_p_value = np.nan

    contrast_summary = {
        "selected_x_category": selected_x_category,
        "selected_y_category": selected_y_category,
        "selected_x_total": selected_x_total,
        "rest_x_total": rest_x_total,
        "selected_y_rate_in_selected_x": safe_float(selected_x_rate),
        "selected_y_rate_in_rest_x": safe_float(rest_x_rate),
        "rate_difference": safe_float(rate_difference),
        "fisher_odds_ratio": fisher_odds_ratio,
        "fisher_p_value": fisher_p_value,
    }

    test_results = pd.DataFrame(
        [
            {
                "test": "Fisher's exact test",
                "statistic": fisher_odds_ratio,
                "statistic_name": "Odds ratio",
                "p_value": fisher_p_value,
                "p_value_display": format_p_value(fisher_p_value),
                "interpretation": interpret_p_value(fisher_p_value),
            },
            {
                "test": "Chi-square test on selected 2x2 contrast",
                "statistic": chi_square_statistic,
                "statistic_name": "Chi-square",
                "p_value": chi_square_p_value,
                "p_value_display": format_p_value(chi_square_p_value),
                "interpretation": interpret_p_value(chi_square_p_value),
            },
        ]
    )

    return {
        "contrast_table": contrast_table,
        "contrast_summary": contrast_summary,
        "test_results": test_results,
    }


def run_categorical_categorical_test_report(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    selected_x_category: str,
    selected_y_category: str,
) -> dict:
    """
    Build a full category-vs-category statistical report.
    """
    crosstab_report = build_categorical_crosstab_report(
        df=df,
        x_column=x_column,
        y_column=y_column,
    )

    selected_contrast = run_selected_category_pair_contrast(
        df=df,
        x_column=x_column,
        y_column=y_column,
        selected_x_category=selected_x_category,
        selected_y_category=selected_y_category,
    )

    return {
        "crosstab_report": crosstab_report,
        "selected_contrast": selected_contrast,
    }