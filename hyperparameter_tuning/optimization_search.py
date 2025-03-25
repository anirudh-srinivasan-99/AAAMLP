from functools import partial
from typing import Tuple

from hyperopt import hp, fmin, tpe, Trials
from hyperopt.pyll.base import scope
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space
from skopt.plots import plot_convergence


def runner(path: str) -> None:
    df = _load_dataset(path)
    features, targets = _get_training_data(df)
    param_search_using_skopt(features, targets)
    param_search_using_hyperopt(features, targets)

def _load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _get_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    features = df.drop(['price_range'], axis=1).values
    target = df['price_range'].values

    return features, target

def param_search_using_skopt(features: np.ndarray, targets: np.ndarray) -> None:
    param_space = [
        # max_depth is an integer between 3 and 10
        space.Integer(3, 15, name="max_depth"),
        # n_estimators is an integer between 50 and 1500
        space.Integer(100, 1500, name="n_estimators"),
        # criterion is a category. here we define list of categories
        space.Categorical(["gini", "entropy"], name="criterion"),
        # you can also have Real numbered space and define a
        # distribution you want to pick it from
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]
    param_names = [
        "max_depth",
        "n_estimators",
        "criterion",
        "max_features"
    ]
    optimization_function = partial(
        _optimize_skopt,
        param_names=param_names,
        features=features,
        targets=targets
    )
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10
    )
    plot_convergence(result)
    # create best params dict and print it
    best_params = dict(zip(
        param_names,
        result.x
    ))
    print(best_params)

def param_search_using_hyperopt(features: np.ndarray, targets: np.ndarray) -> None:
    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 1500, 1)),
        "criterion": hp.choice("criterion", ["gini", "entropy"]),
        "max_features": hp.uniform("max_features", 0, 1)
    }
    optimization_function = partial(
        _optimize_hyperopt,
        features=features,
        targets=targets
    )
    trials = Trials()
    hopt = fmin(
        fn=optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )
    print(hopt)

def _optimize_skopt(param_values, param_names, features: np.ndarray, targets: np.ndarray) -> float:
    params = dict(zip(param_names, param_values))

    rf_model = ensemble.RandomForestClassifier(**params)

    kfv = model_selection.KFold(5) 
    accuracies = []
    for train_idx, test_idx in kfv.split(features):
        train_features = features[train_idx]
        train_targets = targets[train_idx]

        test_features = features[test_idx]
        test_targets = targets[test_idx]

        rf_model.fit(train_features, train_targets)

        target_pred = rf_model.predict(test_features)
        accuracies.append(
            metrics.accuracy_score(test_targets, target_pred)
        )

    return -1 * np.mean(accuracies)

def _optimize_hyperopt(params, features: np.ndarray, targets: np.ndarray) -> float:
    rf_model = ensemble.RandomForestClassifier(**params)

    kfv = model_selection.KFold(5) 
    accuracies = []
    for train_idx, test_idx in kfv.split(features):
        train_features = features[train_idx]
        train_targets = targets[train_idx]

        test_features = features[test_idx]
        test_targets = targets[test_idx]

        rf_model.fit(train_features, train_targets)

        target_pred = rf_model.predict(test_features)
        accuracies.append(
            metrics.accuracy_score(test_targets, target_pred)
        )

    return -1 * np.mean(accuracies)


if __name__ == '__main__':
    PATH = 'hyperparameter_tuning/input/mobile_train.csv'
    runner(PATH)
    plt.show()