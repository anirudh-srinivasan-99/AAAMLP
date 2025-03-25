from typing import Tuple

import numpy as np
import pandas as pd

from sklearn import ensemble, metrics, model_selection, pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def runner(mobile_data_path: str, ):
    df = _load_dataset(mobile_data_path)
    features, target = _get_training_data(df)

    grid_search(features, target)
    random_search(features, target)

def _load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _get_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = df.drop(['price_range'], axis=1).values
    target = df['price_range'].values

    return features, target

def grid_search(features: np.ndarray, target: np.ndarray) -> None:
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }
    rf_model = ensemble.RandomForestClassifier(n_jobs=3)

    best_model = model_selection.GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='accuracy',
        n_jobs=3,
        verbose=10,
        cv=5
    )

    best_model.fit(features, target)
    print(f"Best score: {best_model.best_score_}")
    print("Best parameters set:")
    best_parameters = best_model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")


def random_search(features: np.ndarray, target: np.ndarray) -> None:
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]
    }
    rf_model = ensemble.RandomForestClassifier(n_jobs=3)

    best_model = model_selection.RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring='accuracy',
        n_jobs=3,
        verbose=10,
        cv=5,
    )

    best_model.fit(features, target)
    print(f"Best score: {best_model.best_score_}")
    print("Best parameters set:")
    best_parameters = best_model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
    
def pipeline_grid_search(features: np.ndarray, target: np.ndarray) -> None:
    svd = TruncatedSVD()
    # Initialize the standard scaler
    scl = StandardScaler()
    # We will use SVM here..
    svm_model = SVC()
    clf = pipeline.Pipeline(
        [
            ('svd', svd),
            ('scl', scl),
            ('svm', svm_model)
        ]
    )
    param_grid = {
        'svd__n_components' : [200, 300],
        'svm__C': [10, 12]
    }

    model = model_selection.GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=-1,
        refit=True,
        cv=5
    )
    # Fit Grid Search Model
    model.fit(features, target)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

if __name__ == '__main__':
    PATH = 'hyperparameter_tuning/input/mobile_train.csv'
    runner(PATH)