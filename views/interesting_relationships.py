import pandas as pd
import streamlit as st

from eda.relationship_scanner import build_interesting_relationships_report


def round_display_p_values(df: pd.DataFrame, decimals: int = 5) -> pd.DataFrame:
    """
    Round p-value columns for display only.

    The scanner keeps full-precision numeric values for filtering and sorting.
    """
    output = df.copy()

    p_value_columns = [
        "p_value",
        "p_value_adjusted",
        "secondary_p_value",
    ]

    for column in p_value_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(
                output[column],
                errors="coerce",
            ).round(decimals)

    return output


def filter_and_sort_results(
    df: pd.DataFrame,
    p_value_column: str,
    p_value_threshold: float,
    only_significant: bool,
    top_n: int,
) -> pd.DataFrame:
    """Filter and sort a relationship result table."""
    if df.empty:
        return df

    output = df.copy()

    if only_significant and p_value_column in output.columns:
        output[p_value_column] = pd.to_numeric(
            output[p_value_column],
            errors="coerce",
        )
        output = output[output[p_value_column] <= p_value_threshold]

    sort_columns = []

    if "rank_score" in output.columns:
        sort_columns.append("rank_score")

    if p_value_column in output.columns:
        sort_columns.append(p_value_column)

    if sort_columns:
        ascending = [
            False if column == "rank_score" else True
            for column in sort_columns
        ]
        output = output.sort_values(by=sort_columns, ascending=ascending)

    return output.head(top_n)


def render_result_table(
    title: str,
    df: pd.DataFrame,
    p_value_column: str,
    p_value_threshold: float,
    only_significant: bool,
    top_n: int,
    filename: str,
) -> None:
    """Render a filtered relationship table with a download button."""
    st.markdown(f"#### {title}")

    filtered = filter_and_sort_results(
        df=df,
        p_value_column=p_value_column,
        p_value_threshold=p_value_threshold,
        only_significant=only_significant,
        top_n=top_n,
    )

    if filtered.empty:
        st.info("No results match the current filters.")
        return

    display_df = round_display_p_values(filtered)

    st.dataframe(display_df, width="stretch")

    st.download_button(
        label=f"Download {title}",
        data=display_df.to_csv(index=False),
        file_name=filename,
        mime="text/csv",
    )


def build_all_signals_table(report: dict) -> pd.DataFrame:
    """Combine all relationship tables into one dataframe."""
    result_tables = [
        report["numeric_numeric"],
        report["category_numeric_overall"],
        report["category_numeric_contrasts"],
        report["category_category_overall"],
        report["category_category_contrasts"],
    ]

    non_empty_tables = [
        table for table in result_tables if not table.empty
    ]

    if not non_empty_tables:
        return pd.DataFrame()

    return pd.concat(
        non_empty_tables,
        ignore_index=True,
        sort=False,
    )


def render_scan_summary(report: dict) -> None:
    """Render high-level scan summary metrics."""
    st.subheader("Scan summary")

    summary_cols = st.columns(5)

    summary_cols[0].metric(
        "Numeric columns scanned",
        len(report["candidate_numeric_columns"]),
    )

    summary_cols[1].metric(
        "Categorical columns scanned",
        len(report["candidate_categorical_columns"]),
    )

    summary_cols[2].metric(
        "Numeric-numeric rows",
        len(report["numeric_numeric"]),
    )

    summary_cols[3].metric(
        "Category-numeric contrast rows",
        len(report["category_numeric_contrasts"]),
    )

    summary_cols[4].metric(
        "Category-category rows",
        len(report["category_category_overall"])
        + len(report["category_category_contrasts"]),
    )


def render_interesting_relationships_tab(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> None:
    """Render the Interesting Relationships tab."""
    st.header("Interesting Relationships")

    st.write(
        "Scan the dataset for relationships worth investigating. "
        "Use this as exploratory triage, not as confirmatory evidence."
    )

    with st.expander("Scan controls", expanded=True):
        scan_mode = st.radio(
            "Scan mode",
            options=[
                "Full dataset scan",
                "Target-focused scan",
            ],
            horizontal=True,
        )

        target_column = None

        if scan_mode == "Target-focused scan":
            target_column = st.selectbox(
                "Target column",
                options=df.columns.tolist(),
                help=(
                    "Only relationships involving this target will be scanned. "
                    "The existing scanner logic is reused; the target simply filters "
                    "which checks are run."
                ),
            )

        control_cols_1 = st.columns(4)

        with control_cols_1[0]:
            min_complete_rows = st.number_input(
                "Minimum complete rows",
                min_value=5,
                max_value=10_000,
                value=20,
                step=5,
            )

        with control_cols_1[1]:
            min_group_size = st.number_input(
                "Minimum group size",
                min_value=2,
                max_value=10_000,
                value=20,
                step=5,
            )

        with control_cols_1[2]:
            max_categories = st.number_input(
                "Max categories per column",
                min_value=2,
                max_value=500,
                value=30,
                step=1,
            )

        with control_cols_1[3]:
            max_levels_per_column = st.number_input(
                "Max levels per contrast scan",
                min_value=2,
                max_value=50,
                value=8,
                step=1,
            )

        control_cols_2 = st.columns(4)

        with control_cols_2[0]:
            alpha = st.number_input(
                "Alpha",
                min_value=0.001,
                max_value=0.2,
                value=0.05,
                step=0.001,
                format="%.3f",
            )

        with control_cols_2[1]:
            p_adjust_method = st.selectbox(
                "P-value adjustment",
                options=["fdr_bh", "bonferroni", "holm"],
                index=0,
            )

        with control_cols_2[2]:
            include_category_category_contrasts = st.checkbox(
                "Scan category-category contrasts",
                value=True,
            )

        with control_cols_2[3]:
            max_unique_pct = st.number_input(
                "Max unique % for categorical columns",
                min_value=0.01,
                max_value=1.0,
                value=0.95,
                step=0.01,
                format="%.2f",
            )

        run_scan = st.button("Run relationship scan", type="primary")

    config = {
        "min_complete_rows": min_complete_rows,
        "min_group_size": min_group_size,
        "max_categories": max_categories,
        "max_levels_per_column": max_levels_per_column,
        "max_unique_pct": max_unique_pct,
        "alpha": alpha,
        "p_adjust_method": p_adjust_method,
        "include_category_category_contrasts": include_category_category_contrasts,
        "target_column": target_column,
    }

    if run_scan:
        with st.spinner("Scanning relationships..."):
            st.session_state["interesting_relationships_report"] = (
                build_interesting_relationships_report(
                    df=df,
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    config=config,
                )
            )

            st.session_state["interesting_relationships_scan_mode"] = scan_mode
            st.session_state["interesting_relationships_target_column"] = target_column

    if "interesting_relationships_report" not in st.session_state:
        st.info("Configure the scan, then click **Run relationship scan**.")
        return

    report = st.session_state["interesting_relationships_report"]

    if "warning" in report:
        st.warning(report["warning"])

    scan_mode_used = st.session_state.get(
        "interesting_relationships_scan_mode",
        "Full dataset scan",
    )

    if report.get("target_column") is not None:
        st.info(
            f"Target-focused scan for `{report['target_column']}` "
            f"as a `{report['target_type']}` target."
        )
    else:
        st.info("Full dataset scan: all eligible column pairs were scanned.")

    render_scan_summary(report)

    with st.expander("Columns included in scan"):
        st.markdown("**Numeric columns**")
        st.write(report["candidate_numeric_columns"])

        st.markdown("**Categorical columns**")
        st.write(report["candidate_categorical_columns"])

    st.divider()

    filter_cols = st.columns(4)

    with filter_cols[0]:
        p_value_mode = st.selectbox(
            "Significance filter uses",
            options=["Nominal p-value", "Adjusted p-value"],
            index=0,
        )

    with filter_cols[1]:
        p_value_threshold = st.number_input(
            "P-value threshold",
            min_value=0.001,
            max_value=1.0,
            value=0.05,
            step=0.001,
            format="%.3f",
        )

    with filter_cols[2]:
        only_significant = st.checkbox(
            "Only show significant rows",
            value=True,
        )

    with filter_cols[3]:
        top_n = st.number_input(
            "Top N rows",
            min_value=5,
            max_value=1000,
            value=50,
            step=5,
        )

    p_value_column = (
        "p_value" if p_value_mode == "Nominal p-value" else "p_value_adjusted"
    )

    is_target_scan = scan_mode_used == "Target-focused scan"

    numeric_numeric_title = (
        "Numeric Features vs Numeric Target"
        if is_target_scan and report.get("target_type") == "numeric"
        else "Numeric vs Numeric"
    )

    category_numeric_overall_title = (
        "Numeric Features by Target Category - Overall"
        if is_target_scan and report.get("target_type") == "categorical"
        else "Category vs Numeric - Overall"
    )

    category_numeric_contrast_title = (
        "Numeric Features by Target Category - Contrasts"
        if is_target_scan and report.get("target_type") == "categorical"
        else "Category vs Numeric - Category vs Rest Contrasts"
    )

    category_category_overall_title = (
        "Categorical Features vs Categorical Target - Overall"
        if is_target_scan and report.get("target_type") == "categorical"
        else "Category vs Category - Overall"
    )

    category_category_contrast_title = (
        "Categorical Features vs Categorical Target - Contrasts"
        if is_target_scan and report.get("target_type") == "categorical"
        else "Category vs Category - Category Pair Contrasts"
    )

    relationship_tabs = st.tabs(
        [
            "Numeric vs Numeric",
            "Category vs Numeric",
            "Category vs Category",
            "All Signals",
        ]
    )

    with relationship_tabs[0]:
        render_result_table(
            title=numeric_numeric_title,
            df=report["numeric_numeric"],
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="numeric_numeric_relationships.csv",
        )

    with relationship_tabs[1]:
        render_result_table(
            title=category_numeric_overall_title,
            df=report["category_numeric_overall"],
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="category_numeric_overall_relationships.csv",
        )

        render_result_table(
            title=category_numeric_contrast_title,
            df=report["category_numeric_contrasts"],
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="category_numeric_contrasts.csv",
        )

    with relationship_tabs[2]:
        render_result_table(
            title=category_category_overall_title,
            df=report["category_category_overall"],
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="category_category_overall_relationships.csv",
        )

        render_result_table(
            title=category_category_contrast_title,
            df=report["category_category_contrasts"],
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="category_category_contrasts.csv",
        )

    with relationship_tabs[3]:
        all_results = build_all_signals_table(report)

        render_result_table(
            title="All Relationship Signals",
            df=all_results,
            p_value_column=p_value_column,
            p_value_threshold=p_value_threshold,
            only_significant=only_significant,
            top_n=top_n,
            filename="all_relationship_signals.csv",
        )

    st.caption(
        "These scans run many tests. Nominal p-values are useful for exploration, "
        "but adjusted p-values are safer when reviewing many relationships. "
        "Adjusted p-values are calculated within each result table. Rankings prioritise "
        "effect size or practical difference where possible."
    )