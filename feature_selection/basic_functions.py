import pandas as pd
from sklearn import feature_selection

def _remove_low_variance_features(
    data_df: pd.DataFrame,
    variance_thresh: float
) -> pd.DataFrame:
    var_filter = feature_selection.VarianceThreshold(threshold=variance_thresh)
    return var_filter.fit_transform(data_df)

def _get_corr(
    data_df: pd.DataFrame
) -> pd.DataFrame:
    return data_df.corr()