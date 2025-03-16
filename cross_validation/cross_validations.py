from typing import Tuple
import pandas as pd
from sklearn import model_selection


def _k_fold_cross_validation(df: pd.DataFrame, fold: int, feat_columns: list) -> pd.DataFrame:
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    kfold_obj = model_selection.KFold(n_splits=fold, random_state=1)

    for fold, (_, val) in enumerate(kfold_obj.split(df[feat_columns])):
        df.loc[val, 'kfold'] = fold
    
    return df


def _k_fold_stratifiedcross_validation(
    df: pd.DataFrame, fold: int, target_column: str, feat_columns: list
) -> pd.DataFrame:
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    stratified_kfold_obj = model_selection.StratifiedGroupKFold(n_splits=fold, random_state=1)

    for fold, (_, val) in enumerate(stratified_kfold_obj.split(X=df[feat_columns].values, y=df[target_column].values)):
        df.loc[val, 'kfold'] = fold
    
    return df


def _hold_out_cross_validation(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = model_selection.train_test_split(df, test_size=test_size, random_state=1)
    
    return train_df, test_df


def _leave_one_out_cross_validation(df: pd.DataFrame) -> pd.DataFrame:
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe
    
    loo_obj = model_selection.LeaveOneOut()
    
    for fold, (_, val_idx) in enumerate(loo_obj.split(df)):
        df.loc[val_idx, 'kfold'] = fold
    
    return df


def _group_k_cross_validation(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe
    
    group_kfold_obj = model_selection.GroupKFold(n_splits=fold)
    
    for fold, (train_idx, val_idx) in enumerate(group_kfold_obj.split(df, groups=df[group_column])):
        df.loc[val_idx, 'kfold'] = fold
    
    return df


def _k_fold_validation_for_regression(
    df: pd.DataFrame, folds: int, num_bins: int, target_column: str
) -> pd.DataFrame:
    df.loc[:, 'bins'] = pd.cut(
        df[target_column], bins=num_bins, labels=False
    )
    return _k_fold_stratifiedcross_validation(df, folds, 'bins')

