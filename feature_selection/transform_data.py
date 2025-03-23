import numpy as np
import pandas as pd

def bin_data(
    numeric_col: pd.Series
) -> pd.Series:
    return pd.cut(numeric_col, bins=10)

def log_transform(
    numeric_col: pd.Series
) -> pd.Series:
    return numeric_col.apply(
        lambda val: np.log(val + 1) if val + 1 != 0 else 0
    )