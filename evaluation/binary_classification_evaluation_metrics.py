import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def runner():
    y_true = np.array([0,1,1,1,0,0,0,1])
    y_pred = np.array([0,1,0,1,0,1,0,0])

    y_true_prob = np.array(
        [
            0, 0, 0, 0, 1, 0, 1, 
            0, 0, 1, 0, 1, 0, 0, 1
        ]
    )
    y_pred_prob = np.array(
        [
            0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 
            0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99
        ]
    )
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
    
    print(f'Accuracy Score: {_accuracy(y_true, y_pred):.3f}')
    print(f'Accuracy Score (TP | FP | TN | FN): {_accuracy_using_tp_fp_tn_fn(y_true, y_pred):.3f}')

    print(f'Precision: {_precision_postive_class(y_true, y_pred):.3f}')
    print(f'Recall: {_recall_postive_class(y_true, y_pred):.3f}')
    print(f'F1-Score: {_f1_score_positive_class(y_true, y_pred):.3f}')

    print(f'TPR: {_tpr(y_true, y_pred):.3f}')
    print(f'FPR: {_fpr(y_true, y_pred):.3f}')
    print(f'TNR: {_tnr(y_true, y_pred):.3f}')

    _plot_roc(y_true_prob, y_pred_prob, thresholds)
    print(f'AUC: {_auc_score(y_true_prob, y_pred_prob):.3f}')

    print(f'Log Loss: {_logloss(y_true_prob, y_pred_prob):.3f}')


def _accuracy(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    # return metrics.accuracy_score(y_true, y_pred)
    accurate_pred = []
    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            accurate_pred.append(1)
        else:
            accurate_pred.append(0)
    return np.mean(accurate_pred)


def _accuracy_using_tp_fp_tn_fn(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    tp = _tp_binary(y_true, y_pred)
    fp = _fp_binary(y_true, y_pred)
    tn = _tn_binary(y_true, y_pred)
    fn = _fn_binary(y_true, y_pred)
    return (tp + tn)/ (tp + tn + fp + fn)


def _f1_score_positive_class(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    # return metrics.f1_score(y_true, y_pred)
    p = _precision_postive_class(y_true, y_pred)
    r = _recall_postive_class(y_true, y_pred)
    return 2*(p * r)/ (p + r)


def _plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list
):
    tpr_list, fpr_list = [], []
    for thresh in thresholds:
        # calculate predictions for a given threshold
        temp_pred = [1 if x >= thresh else 0 for x in y_prob]
        # calculate tpr
        temp_tpr = _tpr(y_true, temp_pred)
        # calculate fpr
        temp_fpr = _fpr(y_true, temp_pred)
        # append tpr and fpr to lists
        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)
    
    plt.figure(figsize=(7, 7))
    plt.fill_between(fpr_list, tpr_list, alpha=0.4)
    plt.plot(fpr_list, tpr_list, lw=3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('FPR', fontsize=15)
    plt.ylabel('TPR', fontsize=15)
    plt.show()


def _auc_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
):
    return metrics.roc_auc_score(y_true, y_prob)


def _logloss(
    y_true: np.ndarray,
    y_prob: np.ndarray,
):
    # return metrics.log_loss(y_true, y_prob)
    loss = []
    for yt, yp in zip(y_true, y_prob):
        # Ensures that the probability values (yp) are not exactly 0 or 1, 
        # which would cause issues when computing the logarithm (log(0) is undefined).
        yp = np.clip(yp, 1e-15, 1 - 1e-15)
        loss.append(-((yt)*math.log(yp, math.e) + (1 - yt)*math.log(1 - yp, math.e)))

    return np.mean(loss)

def _tpr(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    return _recall_postive_class(y_true, y_pred)


def _tnr(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    return 1 - _fpr(y_true, y_pred)


def _fpr(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    fp = _fp_binary(y_true, y_pred)
    tn = _tn_binary(y_true, y_pred)

    return fp / (fp + tn)


def _precision_postive_class(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    return metrics.precision_score(y_true, y_pred)
    # tp = _tp_binary(y_true, y_pred)
    # fp = _fp_binary(y_true, y_pred)
    # return (tp)/ (tp + fp)


def _recall_postive_class(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    # return metrics.recall_score(y_true, y_pred)
    tp = _tp_binary(y_true, y_pred)
    fn = _fn_binary(y_true, y_pred)
    return (tp)/ (tp + fn)


def _tp_binary(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    tp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
    
    return tp


def _tn_binary(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 0:
            tn += 1
    
    return tn


def _fp_binary(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    fp = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 0 and yp == 1:
            fp += 1
    
    return fp


def _fn_binary(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> float:
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 0:
            fn += 1
    
    return fn

if __name__ == '__main__':
    runner()


