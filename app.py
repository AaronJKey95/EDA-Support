import pandas as pd
import plotly.express as px
import streamlit as st

from eda.categorical import build_value_counts
from eda.completeness import build_missing_summary, build_row_missing_summary
from eda.correlations import calculate_correlation_matrix, build_correlation_pairs
from eda.load_data import load_csv
from eda.numeric import build_numeric_summary, get_iqr_outlier_summary
from eda.profiling import build_column_profile, get_basic_metadata
from eda.quality import build_quality_checks
from eda.relationships import calculate_pairwise_correlation
from eda.schema import get_categorical_columns, get_numeric_columns

from views.category_numeric_statistics import render_categorical_numeric_statistics
from views.category_category_statistics import render_categorical_categorical_statistics
from views.interesting_relationships import render_interesting_relationships_tab

st.set_page_config(
    page_title="EDA Support App",
    page_icon="📊",
    layout="wide",
)


st.title("📊 EDA Support App")
st.write(
    "Upload a CSV file to generate a first-pass exploratory data analysis report."
)


with st.sidebar:
    st.header("Upload options")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
    )

    separator = st.selectbox(
        "Separator",
        options=[",", ";", "\t", "|"],
        index=0,
    )

    encoding = st.selectbox(
        "Encoding",
        options=["utf-8", "latin1", "cp1252"],
        index=0,
    )

    st.divider()

    st.caption(
        "Designed by ChatGPT, with support from Aaron Key"
    )


if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()


try:
    file_bytes = uploaded_file.getvalue()
    df = load_csv(file_bytes, separator, encoding)
except Exception as error:
    st.error("The file could not be loaded. Try a different separator or encoding.")
    st.exception(error)
    st.stop()


if df.empty:
    st.warning("The uploaded CSV loaded successfully, but it contains no rows.")
    st.stop()


numeric_columns = get_numeric_columns(df)
categorical_columns = get_categorical_columns(df)
column_profile = build_column_profile(df)
metadata = get_basic_metadata(df)


overview_tab, completeness_tab, quality_tab, numeric_tab, categorical_tab, relationships_tab, correlation_tab, interesting_relationships_tab = st.tabs(
    [
        "Overview",
        "Completeness",
        "Data Quality",
        "Numeric Analysis",
        "Categorical Analysis",
        "Relationships",
        "Correlations",
        "Interesting Relationships",
    ]
)

with overview_tab:
    st.header("Dataset overview")

    metric_cols = st.columns(5)

    for col, (label, value) in zip(metric_cols, metadata.items()):
        col.metric(label, value)

    st.subheader("Preview")
    st.dataframe(df.head(100), width="stretch")

    st.subheader("Column profile")
    st.dataframe(column_profile, width="stretch")

    st.subheader("Data types")

    dtype_counts = (
        pd.Series([str(dtype) for dtype in df.dtypes])
        .value_counts()
        .reset_index()
    )
    dtype_counts.columns = ["dtype", "count"]

    st.dataframe(dtype_counts, width="stretch")


with completeness_tab:
    st.header("Completeness")

    missing_summary = build_missing_summary(column_profile)

    st.subheader("Missing values by column")
    st.dataframe(missing_summary, width="stretch")

    columns_with_missing = missing_summary[missing_summary["missing_count"] > 0]

    if columns_with_missing.empty:
        st.success("No missing values detected.")
    else:
        st.subheader("Top missing columns")

        fig = px.bar(
            columns_with_missing.head(30),
            x="column",
            y="missing_pct",
            title="Missing percentage by column",
        )
        fig.update_layout(xaxis_title="Column", yaxis_title="Missing %")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Rows with missing values")

    row_missing_summary = build_row_missing_summary(df)

    st.dataframe(
        row_missing_summary.head(50),
        width="stretch",
    )


with quality_tab:
    st.header("Data quality checks")

    duplicate_rows = int(df.duplicated().sum())
    st.metric("Duplicate rows", duplicate_rows)

    quality_checks = build_quality_checks(df, column_profile)
    st.dataframe(quality_checks, width="stretch")

    st.info(
        "These checks are heuristic. For example, a high-cardinality column may be a valid free-text field, an ID, or a code."
    )


with numeric_tab:
    st.header("Numeric analysis")

    if not numeric_columns:
        st.warning("No numeric columns detected.")
    else:
        st.subheader("Numeric summary statistics")
        numeric_summary = build_numeric_summary(df, numeric_columns)
        st.dataframe(numeric_summary, width="stretch")

        selected_numeric_column = st.selectbox(
            "Select numeric column",
            options=numeric_columns,
        )

        chart_col_1, chart_col_2 = st.columns(2)

        with chart_col_1:
            fig = px.histogram(
                df,
                x=selected_numeric_column,
                title=f"Histogram: {selected_numeric_column}",
                marginal="box",
            )
            st.plotly_chart(fig, width="stretch")

        with chart_col_2:
            fig = px.box(
                df,
                y=selected_numeric_column,
                title=f"Boxplot: {selected_numeric_column}",
            )
            st.plotly_chart(fig, width="stretch")

        st.subheader("Possible outliers using IQR rule")

        outlier_summary = get_iqr_outlier_summary(df, numeric_columns)
        st.dataframe(outlier_summary, width="stretch")


with categorical_tab:
    st.header("Categorical analysis")

    if not categorical_columns:
        st.warning("No categorical columns detected.")
    else:
        selected_categorical_column = st.selectbox(
            "Select categorical column",
            options=categorical_columns,
        )

        top_n = st.slider(
            "Number of categories to show",
            min_value=5,
            max_value=50,
            value=20,
        )

        value_counts = build_value_counts(
            df=df,
            column=selected_categorical_column,
            top_n=top_n,
        )

        st.subheader("Top values")
        st.dataframe(value_counts, width="stretch")

        fig = px.bar(
            value_counts,
            x=selected_categorical_column,
            y="count",
            title=f"Top {top_n} values: {selected_categorical_column}",
        )
        fig.update_layout(xaxis_title=selected_categorical_column, yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")


with relationships_tab:
    st.header("Relationship explorer")

    if df.shape[1] < 2:
        st.warning("At least two columns are needed for relationship analysis.")
    else:
        x_column = st.selectbox("X column", options=df.columns.tolist())
        y_column = st.selectbox("Y column", options=df.columns.tolist(), index=1)

        x_is_numeric = x_column in numeric_columns
        y_is_numeric = y_column in numeric_columns

        if x_column == y_column:
            st.warning("Choose two different columns.")

        elif x_is_numeric and y_is_numeric:
            correlation = calculate_pairwise_correlation(
                df=df,
                x_column=x_column,
                y_column=y_column,
                method="pearson",
            )

            st.metric("Pearson correlation", round(correlation, 4))

            fig = px.scatter(
                df,
                x=x_column,
                y=y_column,
                title=f"{x_column} vs {y_column}",
            )
            st.plotly_chart(fig, width="stretch")

        elif not x_is_numeric and y_is_numeric:
            fig = px.box(
                df,
                x=x_column,
                y=y_column,
                title=f"{y_column} by {x_column}",
            )

            st.plotly_chart(fig, width="stretch")
            render_categorical_numeric_statistics(
                df=df,
                categorical_column=x_column,
                numeric_column=y_column,
            )

        elif x_is_numeric and not y_is_numeric:
            fig = px.box(
                df,
                x=y_column,
                y=x_column,
                title=f"{x_column} by {y_column}",
            )
            st.plotly_chart(fig, width="stretch")
            render_categorical_numeric_statistics(
                df=df,
                categorical_column=y_column,
                numeric_column=x_column,
            )

        else:
            render_categorical_categorical_statistics(
                df=df,
                x_column=x_column,
                y_column=y_column,
            )


with correlation_tab:
    st.header("Correlations")

    if len(numeric_columns) < 2:
        st.warning("At least two numeric columns are needed for correlation analysis.")
    else:
        correlation_method = st.selectbox(
            "Correlation method",
            options=["pearson", "spearman"],
            index=0,
        )

        corr = calculate_correlation_matrix(
            df=df,
            numeric_columns=numeric_columns,
            method=correlation_method,
        )

        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title=f"{correlation_method.title()} correlation matrix",
        )
        st.plotly_chart(fig, width="stretch")

        corr_pairs = build_correlation_pairs(corr)

        st.subheader("Strongest relationships")
        st.dataframe(
            corr_pairs.head(20),
            width="stretch",
        )

with interesting_relationships_tab:
    render_interesting_relationships_tab(
        df=df,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
