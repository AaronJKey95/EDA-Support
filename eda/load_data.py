import io

import pandas as pd
import streamlit as st


@st.cache_data
def load_csv(file_bytes: bytes, separator: str, encoding: str) -> pd.DataFrame:
    """
    Load uploaded CSV bytes into a pandas DataFrame.

    Streamlit reruns the script whenever widgets change, so caching avoids
    repeatedly parsing the same uploaded file.
    """
    return pd.read_csv(
        io.BytesIO(file_bytes),
        sep=separator,
        encoding=encoding,
    )