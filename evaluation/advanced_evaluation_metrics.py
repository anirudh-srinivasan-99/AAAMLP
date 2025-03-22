import numpy as np
from sklearn import metrics

def runner():
    y_true_multi = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
    y_pred_multi = np.array([2, 1, 3, 1, 2, 3, 3, 1, 2])

    y_true_binary = np.array([0,1,1,1,0,0,0,1])
    y_pred_binary = np.array([0,1,0,1,0,1,0,0])

    print(f'Cohen-Kappa Score: {_cohen_kappa_score(y_true_multi, y_pred_multi):.3f}')
    print(f'MCC Score: {_mcc(y_true_binary, y_pred_binary):.3f}')

def  _cohen_kappa_score(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic')

def  _mcc(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return metrics.matthews_corrcoef(y_true, y_pred)

if __name__ == '__main__':
    runner()