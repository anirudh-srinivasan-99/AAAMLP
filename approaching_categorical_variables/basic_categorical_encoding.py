from typing import List

import numpy as np
import pandas as pd
from sklearn import preprocessing

def runner(train_path: str, test_path: str):
    df_train = _load_dataframe(train_path)
    df_test = _load_dataframe(test_path)


def _load_dataframe(path) -> pd.DataFrame:
    return pd.read_csv(path)


def _label_encoding(
    categorical_column: pd.Series,
) -> pd.Series:
    # Assings an Integer from 1 to N for N unique classes.
    # Note that filling the Missing Values as LabelEncoder
    # does not process NaNs.
    categorical_column = categorical_column.fillna('None')
    lbl_enc = preprocessing.LabelEncoder()

    return lbl_enc.fit_transform(categorical_column)


def _one_hot_encoding(
   categorical_column: pd.Series,
) -> pd.Series:
    categorical_column = categorical_column.fillna('None')
    one_hot_enc = preprocessing.OneHotEncoder(sparse=False)

    return one_hot_enc.fit_transform(categorical_column)


def _count_encoding(
     categorical_column: pd.Series,
) -> pd.Series:
    categorical_column = categorical_column.fillna('None')
    count_mapping = categorical_column.value_counts()

    return categorical_column.map(count_mapping)

def _combine_categorical_variables(
    columns_list: List[pd.Series],
    separator: str = '_'
) -> pd.Series:
    concatenated_column = columns_list[0].astype(str)
    
    for series in columns_list[1:]:
        concatenated_column = concatenated_column + separator + series.astype(str)
    
    return concatenated_column


def _rare_and_unknown_encoding(
    categorical_column: pd.Series,
    rare_count_threshold: int # based on manual inspection of the data.
) -> pd.Series:
    # Fills the empty values as UNKNOWN and the categories with
    # count less than the rare_count_threshold as RARE
    categorical_column = categorical_column.fillna('UNKNOWN')
    value_counts = categorical_column.value_counts()
    categorical_column = categorical_column.apply(
        lambda x: 'RARE' if value_counts.get(x, 0) < rare_count_threshold else x
    )

    return categorical_column


def _mean_target_encoding(
    data_df: pd.DataFrame,
    cat_col_name: str,
    target_col_name: str
) -> pd.Series:
    # Cleaning the dataframe to only contain the categorical column
    # to be encoded and the target column
    data_df = data_df[[cat_col_name, target_col_name]]

    # Handling NaNs
    data_df.loc[:, cat_col_name] = data_df[cat_col_name].fillna('None')

    # Aggregating with categorical column and finding the mean of target_col_name
    # for each unique category.
    mean_encoded = data_df.groupby(cat_col_name).aggregate({target_col_name: 'mean'})

    # Mapping the encoded target column into the categorical column.
    encoded_column = data_df[cat_col_name].map(mean_encoded[target_col_name])

    return encoded_column

if __name__ == '__main__':
    TRAIN_PATH = 'approaching_categorical_variables/inputs/cat_train.csv'
    TEST_PATH = 'approaching_categorical_variables/inputs/cat_test.csv'
    runner(TRAIN_PATH, TEST_PATH)