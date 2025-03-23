from typing import Dict

import numpy as np
import pandas as pd
from sklearn import preprocessing

def get_feat_stats(
    x: np.ndarray
) -> Dict[str, float]:
    stat_dict = {
        'mean': np.mean(x),
        'max': np.max(x),
        'min': np.min(x),
        'std': np.std(x),
        'var': np.var(x),
        'ptp': np.ptp(x),
        'percentile_10': np.percentile(x, 10),
        'percentile_50': np.percentile(x, 50),
        'percentile_90': np.percentile(x, 90),

        'quartile_10': np.quantile(x, 10),
        'quartile_50': np.quantile(x, 50),
        'quartile_90': np.quantile(x, 90),
    }

    return stat_dict

def get_poly_feats(
    df: pd.DataFrame
) -> pd.DataFrame:
    pf = preprocessing.PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=False
    )
    # fit to the features
    pf.fit(df)
    # create polynomial features
    poly_feats = pf.transform(df)
    # create a dataframe with all the features
    num_feats = poly_feats.shape[1]
    df_transformed = pd.DataFrame(
        poly_feats,
        columns=[f"f_{i}" for i in range(1, num_feats + 1)]
    )
    return df_transformed