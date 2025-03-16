from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics, tree

from cross_validation.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN, TARGET_VALUE_MAPPING,
    WINE_DATASET_PATH
)
def runner():
    df = _load_data()
    df_train, df_test = _train_test_split(df)
    train_scores, test_scores = _get_train_test_eval_score_over_max_depth(
        df_train, df_test
    )
    _plot_overfit(train_scores, test_scores)


def _load_data() -> pd.DataFrame:
    df = pd.read_csv(WINE_DATASET_PATH)
    df.loc[:, TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_VALUE_MAPPING)
    return df

def _train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1).reset_index(drop=True)
    df_train, df_test = df.head(1000), df.tail(599)
    # Raises an assertion Error if the number of rows in df < the sum of rows
    # in train and testing.
    # This is to ensure that there isnt any data leak between training
    # and testing data.
    assert df.shape[0] >= df_train.shape[0] + df_test.shape[0]
    return df_train, df_test

def _train_dt_model(df_train: pd.DataFrame, max_depth: int) -> tree.DecisionTreeClassifier:
    dt_model = tree.DecisionTreeClassifier(max_depth=max_depth)

    dt_model.fit(
        df_train[FEATURE_COLUMNS].values,
        df_train[TARGET_COLUMN]
    )
    return dt_model

def _get_train_test_eval_score_over_max_depth(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[list, list]:
    train_scores, test_scores = [0.5], [0.5]
    for max_depth in range(1, 26):
        dt_model = _train_dt_model(df_train, max_depth)
        train_scores.append(_get_evaluation_score(dt_model, df_train))
        test_scores.append(_get_evaluation_score(dt_model, df_test))
    return train_scores, test_scores

def _get_evaluation_score(
    dt_model: tree.DecisionTreeClassifier,
    df: pd.DataFrame
) -> float:
    predictions = dt_model.predict(df[FEATURE_COLUMNS])
    score = metrics.accuracy_score(df[TARGET_COLUMN], predictions)

    return score

def _plot_overfit(train_scores: list, test_scores: list) -> bool:
    plt.figure(figsize=(10, 5))
    sns.set_style("whitegrid")
    plt.plot(train_scores, label="Train Scores")
    plt.plot(test_scores, label="Test Scores")
    plt.legend(loc="upper left", prop={'size': 15})
    plt.xticks(range(0, 26, 5))
    plt.xlabel("max_depth", size=20)
    plt.ylabel("accuracy", size=20)
    plt.show()
    return True

if __name__ == '__main__':
    runner()