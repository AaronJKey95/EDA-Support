import pandas as pd
import streamlit as st

from eda.statistical_tests import (
    format_p_value,
    prepare_categorical_categorical_data,
    run_categorical_categorical_test_report,
)


def format_metric_value(value, digits: int = 4):
    """Format numeric values for Streamlit metric cards."""
    if pd.isna(value):
        return "N/A"

    return round(float(value), digits)


def render_categorical_categorical_statistics(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> None:
    """
    Render statistical tests for categorical-vs-categorical relationships.
    """
    st.subheader("Categorical relationship tests")

    working_df = prepare_categorical_categorical_data(
        df=df,
        x_column=x_column,
        y_column=y_column,
    )

    if working_df.empty:
        st.warning("No complete rows are available for categorical relationship testing.")
        return

    x_counts = working_df[x_column].value_counts()
    y_counts = working_df[y_column].value_counts()

    x_options = x_counts.index.tolist()
    y_options = y_counts.index.tolist()

    if len(x_options) < 2 or len(y_options) < 2:
        st.warning("At least two categories are required in both columns.")
        return

    if len(x_options) > 50 or len(y_options) > 50:
        st.warning(
            "One or both columns have more than 50 categories. "
            "The tests may be noisy or hard to interpret."
        )

    selector_cols = st.columns(2)

    with selector_cols[0]:
        selected_x_category = st.selectbox(
            f"Select {x_column} category",
            options=x_options,
            format_func=lambda value: f"{value} (n={x_counts.get(value, 0)})",
            key=f"selected_x_category_{x_column}_{y_column}",
        )

    with selector_cols[1]:
        selected_y_category = st.selectbox(
            f"Select {y_column} category",
            options=y_options,
            format_func=lambda value: f"{value} (n={y_counts.get(value, 0)})",
            key=f"selected_y_category_{x_column}_{y_column}",
        )

    report = run_categorical_categorical_test_report(
        df=df,
        x_column=x_column,
        y_column=y_column,
        selected_x_category=selected_x_category,
        selected_y_category=selected_y_category,
    )

    crosstab_report = report["crosstab_report"]

    if "error" in crosstab_report:
        st.warning(crosstab_report["error"])
        return

    st.markdown("#### Observed counts")
    st.dataframe(crosstab_report["observed"], width="stretch")

    st.markdown("#### Row percentages")
    st.dataframe(crosstab_report["row_percentages"].round(2), width="stretch")

    st.markdown("#### Overall association tests")
    st.dataframe(crosstab_report["overall_tests"], width="stretch")

    st.markdown("#### Association effect sizes")
    st.dataframe(crosstab_report["association_summary"], width="stretch")

    expected_summary = crosstab_report["expected_count_summary"]

    summary_cols = st.columns(4)

    summary_cols[0].metric(
        "Minimum expected count",
        format_metric_value(expected_summary["minimum_expected_count"]),
    )

    summary_cols[1].metric(
        "Cells expected < 5",
        expected_summary["cells_with_expected_count_below_5"],
    )

    summary_cols[2].metric(
        "Total cells",
        expected_summary["total_cells"],
    )

    summary_cols[3].metric(
        "% cells expected < 5",
        format_metric_value(expected_summary["pct_cells_with_expected_count_below_5"]),
    )

    if expected_summary["cells_with_expected_count_below_5"] > 0:
        st.warning(
            "Some expected cell counts are below 5. "
            "The chi-square approximation may be less reliable, especially with sparse tables."
        )

    fisher_result = crosstab_report["fisher_exact_result"]

    if fisher_result is not None:
        st.markdown("#### Fisher's exact test")

        fisher_display = pd.DataFrame([fisher_result])
        st.dataframe(fisher_display, width="stretch")

        fisher_p = fisher_result["p_value"]

        if not pd.isna(fisher_p) and fisher_p < 0.05:
            st.success(
                f"Fisher's exact test suggests a statistically significant association. "
                f"p = {format_p_value(fisher_p)}."
            )
        else:
            st.info(
                f"Fisher's exact test does not suggest a statistically significant association. "
                f"p = {format_p_value(fisher_p)}."
            )

    with st.expander("Expected counts"):
        st.dataframe(crosstab_report["expected"].round(2), width="stretch")

    with st.expander("Standardized residuals"):
        st.write(
            "Larger absolute residuals suggest cells contributing more strongly to the chi-square result."
        )
        st.dataframe(
            crosstab_report["standardized_residuals"].round(2),
            width="stretch",
        )

    selected_contrast = report["selected_contrast"]

    st.markdown(
        f"#### Selected contrast: {selected_x_category} vs rest, "
        f"{selected_y_category} vs rest"
    )

    if "error" in selected_contrast:
        st.warning(selected_contrast["error"])
        return

    contrast_table = selected_contrast["contrast_table"]
    contrast_summary = selected_contrast["contrast_summary"]
    contrast_tests = selected_contrast["test_results"]

    st.dataframe(contrast_table, width="stretch")

    contrast_metric_cols = st.columns(5)

    contrast_metric_cols[0].metric(
        f"{selected_y_category} rate in {selected_x_category}",
        format_metric_value(
            contrast_summary["selected_y_rate_in_selected_x"] * 100
        ),
    )

    contrast_metric_cols[1].metric(
        f"{selected_y_category} rate in rest",
        format_metric_value(
            contrast_summary["selected_y_rate_in_rest_x"] * 100
        ),
    )

    contrast_metric_cols[2].metric(
        "Rate difference",
        format_metric_value(
            contrast_summary["rate_difference"] * 100
        ),
    )

    contrast_metric_cols[3].metric(
        "Odds ratio",
        format_metric_value(contrast_summary["fisher_odds_ratio"]),
    )

    contrast_metric_cols[4].metric(
        "Fisher p-value",
        format_p_value(contrast_summary["fisher_p_value"]),
    )

    st.dataframe(contrast_tests, width="stretch")

    fisher_p = contrast_summary["fisher_p_value"]

    if not pd.isna(fisher_p) and fisher_p < 0.05:
        st.success(
            f"The selected 2x2 contrast suggests a statistically significant association. "
            f"p = {format_p_value(fisher_p)}."
        )
    else:
        st.info(
            f"The selected 2x2 contrast does not suggest a statistically significant association. "
            f"p = {format_p_value(fisher_p)}."
        )

    st.caption(
        "Interpretation note: the overall chi-square test tells you whether the two categorical "
        "variables appear associated overall. It does not tell you which specific cells drive the "
        "association. Use the standardized residuals, percentages, Cramér's V, and selected 2x2 "
        "contrast alongside the p-values."
    )