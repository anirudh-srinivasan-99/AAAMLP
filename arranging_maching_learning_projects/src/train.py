import argparse
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from typing import Tuple

from arranging_maching_learning_projects.src.config import MODEL_FOLDER, TARGET_COL_NAME, TRAINING_DATASET
from arranging_maching_learning_projects.src.model_dispatcher import models


def runner(fold: int, model_name: str) -> None:
    df = _load_training_dataset(TRAINING_DATASET)
    df_train, df_val = _get_train_validation(df, fold)

    fitted_model = _train_model(df_train, model_name)
    evaluation_score = _evaluate_model(df_val, fitted_model)
    _save_model(fitted_model, model_name, fold)

    print(f'Fold: {fold}, Model: {model_name}, Evaluation Score: {evaluation_score:.3f}')


def _load_training_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _get_train_validation(df: pd.DataFrame, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train = df[df['kfolds'] != fold]
    df_val = df[df['kfolds'] == fold]

    return df_train, df_val


def _get_feature_target(df: pd.DataFrame, target_col_name: str) -> Tuple[np.ndarray, np.ndarray]:
    feats = df.drop(target_col_name, axis = 1).values
    target = df[target_col_name].values
    
    return feats, target


def _train_model(df_train: pd.DataFrame, model_name: str) -> sklearn.base.BaseEstimator:
    train_feats, train_targets = _get_feature_target(df_train, TARGET_COL_NAME)
    model = models[model_name]
    model.fit(train_feats, train_targets)
    return model


def _evaluate_model(df_val: pd.DataFrame, model: sklearn.base.BaseEstimator) -> float:
    val_feats, val_targets = _get_feature_target(df_val, TARGET_COL_NAME)
    
    model_targets = model.predict(val_feats)
    evaluation_score = metrics.accuracy_score(val_targets, model_targets)
    return evaluation_score


def _save_model(model: sklearn.base.BaseEstimator, model_name: str, fold: int) -> bool:
    save_path = os.path.join(
        MODEL_FOLDER, f'{model_name}_{fold}.bin'
    )
    joblib.dump(model, save_path)
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    args = parser.parse_args()

    if args.model not in models:
        raise KeyError(f'Expects only {list(models.keys())}, but got {args.model}.')

    runner(
        fold=args.fold,
        model_name=args.model
    )