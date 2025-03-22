from typing import List

import numpy as np
import sklearn.metrics as metrics


def runner():
    y_true = [
        [1, 2, 3],
        [0, 2],
        [1],
        [2, 3],
        [1, 0],
        []
    ]

    y_pred = [
        [0, 1, 2],
        [1],
        [0, 2, 3],
        [2, 3, 4, 0],
        [0, 1, 2],
        [0]
    ]

    print(f'PK Score: {_pk(y_true[0], y_pred[0], 2):.3f}')
    print(f'PK Score: {_pk(y_true[1], y_pred[1], 2):.3f}')
    print(f'PK Score: {_pk(y_true[2], y_pred[2], 2):.3f}')

    print(f'APK Score: {_apk(y_true[0], y_pred[0], 3):.3f}')
    print(f'APK Score: {_apk(y_true[1], y_pred[1], 3):.3f}')
    print(f'APK Score: {_apk(y_true[2], y_pred[2], 3):.3f}')

    print(f'MAPK Score: {_mapk(y_true, y_pred, 1):.3f}')


def _pk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int
) -> float:
    top_k_true = set(y_true[:k])
    top_k_pred = set(y_pred[:k])

    return len(top_k_pred.intersection(top_k_true))/ k


def _apk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int
) -> float:
    pk_list = [_pk(y_true, y_pred, i) for i in range(1, k + 1)]

    return np.mean(pk_list)


def _mapk(
    y_trues: List[np.ndarray],
    y_preds: List[np.ndarray],
    k: int
) -> float:
    apks = []
    for y_true, y_pred in zip(y_trues, y_preds):
        apks.append(_apk(y_true, y_pred, k))

    return np.mean(apks)


if __name__ == '__main__':
    runner()