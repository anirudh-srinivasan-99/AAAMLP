from typing import Union

import numpy as np
import pandas as pd
from sklearn import feature_selection


class UnivariateFeatureSelection:
    def __init__(
        self,
        n_features: Union[int, float],
        problem_type: str,
        scoring: str
    ):
        self.feature_selection_algo = None
        if problem_type == 'classification':
            valid_scoring = {
                'mutual_info_classif': feature_selection.mutual_info_classif,
                'chi2': feature_selection.chi2,
                'f_classif': feature_selection.f_classif
            }
        elif problem_type == 'regression':
            valid_scoring = {
                'mutual_info_regression': feature_selection.mutual_info_regression,
                'f_regression': feature_selection.f_regression
            }
        else:
            raise ValueError
        
        if scoring not in valid_scoring:
            raise ValueError

        # Use SelectKBest if n_features is int 
        # Use SelectPercentile if n_features is float
        if isinstance(n_features, int):
            self.feature_selection_algo = feature_selection.SelectKBest(
                score_func=valid_scoring[scoring],
                k=n_features,
            )
        elif isinstance(n_features, float):
            if n_features > 1:
                raise ValueError

            self.feature_selection_algo = feature_selection.SelectPercentile(
                score_func=valid_scoring[scoring],
                percentile=int(n_features * 100),
            )
        else:
            raise TypeError
        

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.feature_selection_algo.fit(X, y)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.feature_selection_algo.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.feature_selection_algo.fit_transform(X, y)


