import pandas as pd
import streamlit as st

from eda.statistical_tests import (
    format_p_value,
    run_categorical_numeric_test_report,
)


def format_metric_value(value, digits: int = 4):
    """Format numeric values for Streamlit metric cards."""
    if pd.isna(value):
        return "N/A"

    return round(float(value), digits)


def render_categorical_numeric_statistics(
    df: pd.DataFrame,
    categorical_column: str,
    numeric_column: str,
) -> None:
    """
    Render statistical tests for categorical vs numeric relationships.

    This is Streamlit UI logic. The underlying calculations live in
    eda/statistical_tests.py.
    """
    st.subheader("Statistical relationship tests")

    complete_rows = df[[categorical_column, numeric_column]].dropna().copy()

    if complete_rows.empty:
        st.warning("No complete rows are available for statistical testing.")
        return

    complete_rows[categorical_column] = complete_rows[categorical_column].astype(
        "string"
    )

    category_counts = complete_rows[categorical_column].value_counts()
    category_options = category_counts.index.tolist()
    category_counts_lookup = category_counts.to_dict()

    if len(category_options) < 2:
        st.warning("At least two categories are required for statistical testing.")
        return

    if len(category_options) > 50:
        st.warning(
            "This column has more than 50 categories. Tests may be noisy, slow, or difficult to interpret."
        )

    alpha = st.number_input(
        "Significance level alpha",
        min_value=0.001,
        max_value=0.2,
        value=0.05,
        step=0.001,
        format="%.3f",
        key=f"alpha_{categorical_column}_{numeric_column}",
    )

    selected_category = st.selectbox(
        "Compare one category against the rest",
        options=category_options,
        format_func=lambda category: (
            f"{category} (n={category_counts_lookup.get(category, 0)})"
        ),
        key=f"selected_category_{categorical_column}_{numeric_column}",
    )

    report = run_categorical_numeric_test_report(
        df=df,
        categorical_column=categorical_column,
        numeric_column=numeric_column,
        selected_category=selected_category,
        alpha=alpha,
    )

    st.markdown("#### Group summary")
    st.dataframe(report["group_summary"], width="stretch")

    st.markdown("#### Overall difference across all categories")
    st.dataframe(report["overall_tests"], width="stretch")

    selected_vs_rest = report["selected_vs_rest"]

    if "error" in selected_vs_rest:
        st.warning(selected_vs_rest["error"])
        return

    contrast = selected_vs_rest["contrast_summary"]
    test_results = selected_vs_rest["test_results"]

    st.markdown(f"#### {selected_category} vs remainder of cohort")

    metric_cols = st.columns(6)

    metric_cols[0].metric("Selected n", contrast["selected_n"])
    metric_cols[1].metric("Rest n", contrast["rest_n"])

    metric_cols[2].metric(
        "Mean difference",
        format_metric_value(contrast["mean_difference"]),
    )

    metric_cols[3].metric(
        "Median difference",
        format_metric_value(contrast["median_difference"]),
    )

    metric_cols[4].metric(
        "Cohen's d",
        format_metric_value(contrast["cohens_d"]),
    )

    metric_cols[5].metric(
        "Approx. power",
        format_metric_value(contrast["approximate_ttest_power"]),
    )

    contrast_summary_table = pd.DataFrame(
        [
            {"measure": "Selected category", "value": contrast["selected_category"]},
            {"measure": "Selected n", "value": contrast["selected_n"]},
            {"measure": "Rest n", "value": contrast["rest_n"]},
            {"measure": "Selected mean", "value": contrast["selected_mean"]},
            {"measure": "Rest mean", "value": contrast["rest_mean"]},
            {"measure": "Mean difference", "value": contrast["mean_difference"]},
            {"measure": "Selected median", "value": contrast["selected_median"]},
            {"measure": "Rest median", "value": contrast["rest_median"]},
            {"measure": "Median difference", "value": contrast["median_difference"]},
            {"measure": "Cohen's d", "value": contrast["cohens_d"]},
            {
                "measure": "Rank-biserial correlation",
                "value": contrast["rank_biserial_correlation"],
            },
            {
                "measure": "Approximate t-test power",
                "value": contrast["approximate_ttest_power"],
            },
        ]
    )

    st.dataframe(contrast_summary_table, width="stretch")

    st.markdown("#### Selected category vs rest tests")
    st.dataframe(test_results, width="stretch")

    welch_row = test_results[test_results["test"] == "Welch t-test"].iloc[0]
    mann_whitney_row = test_results[
        test_results["test"] == "Mann-Whitney U"
    ].iloc[0]

    welch_p = welch_row["p_value"]
    mann_whitney_p = mann_whitney_row["p_value"]

    if not pd.isna(welch_p) and welch_p < alpha:
        st.success(
            f"Welch t-test: {selected_category} appears statistically different from "
            f"the remainder of the cohort. p = {format_p_value(welch_p)}."
        )
    else:
        st.info(
            f"Welch t-test: {selected_category} does not appear statistically different "
            f"from the remainder of the cohort. p = {format_p_value(welch_p)}."
        )

    if not pd.isna(mann_whitney_p) and mann_whitney_p < alpha:
        st.success(
            f"Mann-Whitney U: {selected_category} appears statistically different from "
            f"the remainder of the cohort. p = {format_p_value(mann_whitney_p)}."
        )
    else:
        st.info(
            f"Mann-Whitney U: {selected_category} does not appear statistically different "
            f"from the remainder of the cohort. p = {format_p_value(mann_whitney_p)}."
        )

    st.caption(
        "Interpretation note: p-values show evidence against a null hypothesis, not practical importance. "
        "Use the effect sizes, group summaries, and plots alongside the p-values. "
        "The power value is approximate and based on the observed Cohen's d."
    )