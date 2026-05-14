# Streamlit EDA App

A modular Streamlit application for exploratory data analysis of uploaded CSV files, built with LLM support (ChatGPT 5.5)

The app is designed to help analysts and data scientists quickly understand a dataset by profiling completeness, data quality, distributions, statistical relationships, correlations, and target-focused relationship signals.

## Features

### CSV Upload and Dataset Overview

- Upload CSV files through the Streamlit interface
- Choose separator and encoding options
- Preview the uploaded dataset
- View row count, column count, duplicate rows, missing values, and memory usage
- Inspect inferred data types and column-level profile information

### Completeness Analysis

- Missing values by column
- Missing percentage by column
- Row-level missingness summary
- Visual summary of columns with missing values

### Data Quality Checks

- Duplicate row count
- Constant column detection
- High-cardinality column detection
- Possible ID column detection
- Basic warnings for columns that may need further inspection

### Numeric Analysis

- Descriptive statistics for numeric columns
- Histograms
- Boxplots
- IQR-based possible outlier detection

### Categorical Analysis

- Frequency tables
- Top category visualisations
- High-level category distribution exploration

### Relationship Explorer

The app supports interactive exploration between selected columns.

Supported relationship types include:

- Numeric vs numeric
  - Scatter plots
  - Pearson correlation

- Categorical vs numeric
  - Boxplots by category
  - Group summaries
  - Kruskal-Wallis test
  - One-way ANOVA
  - Category-vs-rest comparisons
  - Mann-Whitney U test
  - Welch t-test
  - Cohen’s d
  - Rank-biserial correlation
  - Approximate observed power

- Categorical vs categorical
  - Crosstabs
  - Row percentages
  - Chi-square test of independence
  - Cramér’s V
  - Tschuprow’s T
  - Fisher’s exact test for 2x2 tables
  - Expected counts
  - Standardised residuals
  - Category-pair contrast testing

### Interesting Relationships Scanner

The app can scan the dataset for relationships worth investigating.

It produces ranked tables for:

- Numeric vs numeric relationships
- Category vs numeric overall relationships
- Category vs numeric category-vs-rest contrasts
- Category vs category overall associations
- Category vs category contrast tests
- Combined relationship signals

The scanner supports:

- Nominal p-value filtering
- Adjusted p-value filtering
- Benjamini-Hochberg FDR adjustment
- Bonferroni adjustment
- Holm adjustment
- Minimum group size controls
- Maximum category controls
- Target-focused scanning

### Target-Focused Relationship Scanning

Users can optionally select a target column.

When a target is selected, the relationship scanner only runs checks involving that target. This is useful for modelling-style EDA where the user wants to quickly understand which features are most associated with an outcome.

## Project Structure

```text
Streamlit_EDA_App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── eda/
│   ├── __init__.py
│   ├── categorical.py
│   ├── completeness.py
│   ├── correlations.py
│   ├── load_data.py
│   ├── numeric.py
│   ├── profiling.py
│   ├── quality.py
│   ├── relationship_scanner.py
│   ├── relationships.py
│   ├── schema.py
│   └── statistical_tests.py
│
└── views/
    ├── __init__.py
    ├── category_category_statistics.py
    ├── formatting.py
    ├── interesting_relationships.py
    └── relationship_statistics.py
```

## Design Principles

The project separates analysis logic from Streamlit interface logic.

```text
eda/      Pure data analysis and statistical logic
views/    Streamlit rendering logic
app.py    Main application entry point
```

This makes the project easier to maintain, test, and extend.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/streamlit-eda-app.git
cd streamlit-eda-app
```

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

From the project root, run:

```bash
streamlit run app.py
```

The app should open in your browser.

## Requirements

The app currently uses:

```text
streamlit
pandas
numpy
plotly
scipy
statsmodels
```

## Example Workflow

1. Start the app with `streamlit run app.py`
2. Upload a CSV file
3. Review the dataset overview
4. Check completeness and data quality warnings
5. Explore numeric and categorical distributions
6. Use the Relationship Explorer for specific column comparisons
7. Use Interesting Relationships to scan for relationships worth investigating
8. Optionally choose a target column for target-focused relationship scanning

## Statistical Notes

This app is designed for exploratory analysis.

The statistical outputs should be treated as prompts for further investigation rather than final confirmatory evidence.

Important caveats:

- Running many tests increases the risk of false positives
- Nominal p-values are not corrected for multiple testing
- Adjusted p-values are provided in the relationship scanner
- Large datasets can make very small differences statistically significant
- Effect sizes should be considered alongside p-values
- Sparse categorical tables can make chi-square results unreliable
- Correlation does not imply causation
- The approximate power calculation is exploratory and based on observed effect size

## Data Privacy Notes

When run locally, uploaded files are processed within the local Streamlit session.

Do not commit real datasets, sensitive data, patient data, or local secrets to GitHub.

Recommended `.gitignore` entries include:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/
venv/
env/
*.egg-info/

# Streamlit
.streamlit/secrets.toml

# Environment files
.env
.env.*
!.env.example

# Data files
*.csv
*.xlsx
*.xls
*.parquet
*.feather
*.db
*.sqlite

# Outputs
outputs/
reports/
exports/

# OS/editor files
.DS_Store
Thumbs.db
.vscode/
.idea/
```

## Current Status

This project is currently an active prototype.

Implemented:

- CSV upload
- Dataset overview
- Completeness analysis
- Data quality checks
- Numeric profiling
- Categorical profiling
- Relationship explorer
- Statistical tests for categorical/numeric relationships
- Statistical tests for categorical/categorical relationships
- Interesting relationship scanner
- Target-focused relationship scanning

Planned improvements:

- Better semantic column role detection
- Data Quality Findings summary page
- Effect-size interpretation labels
- Click-through from relationship scanner into relationship explorer
- Exportable EDA reports
- Dataset comparison and drift checks
- Optional data cleaning suggestions

## Development Notes

Before committing changes, run:

```bash
python -m compileall .
```

Then run the app:

```bash
streamlit run app.py
```

Basic Git workflow:

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

## Disclaimer

This tool is intended to support exploratory data analysis. It should not be used as the sole basis for clinical, operational, financial, or policy decisions without appropriate review and validation.
