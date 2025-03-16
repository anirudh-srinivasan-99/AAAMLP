import pandas as pd
from sklearn import model_selection

from arranging_maching_learning_projects.src.config import INPUT_DATASET, TRAINING_DATASET

def _create_k_folds(df: pd.DataFrame) -> pd.DataFrame:
    df['kfolds'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kfolds = model_selection.KFold(n_splits=5)
    for fold, (_, val) in enumerate(kfolds.split(X=df)):
        df.loc[val, 'kfolds'] = fold
    return df

if __name__ == '__main__':
    df = pd.read_csv(INPUT_DATASET)
    df_k_folds = _create_k_folds(df)
    df_k_folds.to_csv(TRAINING_DATASET)