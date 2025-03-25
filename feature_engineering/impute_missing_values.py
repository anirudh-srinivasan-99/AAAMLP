import numpy as np
import pandas as pd
from sklearn import impute


def fill_with_0(
    numeric_col: pd.Series
) -> pd.Series:
    return numeric_col.fillna(0)


def fill_with_central_tendency(
    numeric_col: pd.Series,
    central_tendency: str
) -> pd.Series:
    mean = numeric_col.mean()
    median = numeric_col.median()
    mode = numeric_col.mode().to_list()[0]

    if central_tendency == 'mean':
        return numeric_col.fillna(mean)
    
    elif central_tendency == 'median':
        return numeric_col.fillna(median)
    
    elif central_tendency == 'mode':
        return numeric_col.fillna(mode)

    else:
        raise ValueError()


def fill_with_knn_imputer(
    numeric_cols: np.ndarray, # 2d array of missing values
) -> np.ndarray:
    knn_imputer = impute.KNNImputer(n_neighbors=1)
    return knn_imputer.fit_transform(numeric_cols)


if __name__ == '__main__':
    X = np.random.randint(1, 15, (10, 6))
    # convert the array to float
    X = X.astype(float)
    # randomly assign 10 elements to NaN (missing)
    X.ravel()[np.random.choice(X.size, 10, replace=False)] = np.nan
    print('BEFORE FILL !!')
    print(X)
    print('AFTER FILL !!')
    print(fill_with_knn_imputer(X))
        