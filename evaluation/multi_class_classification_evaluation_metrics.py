from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics

from evaluation.binary_classification_evaluation_metrics import (
    _precision_postive_class, _recall_postive_class,
    _tp_binary, _fp_binary, _fn_binary
)


def runner():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2])
    y_pred = np.array([0, 2, 1, 0, 2, 1, 0, 0, 2])

    print(f'Macro Precision: {_macro_precision(y_true, y_pred):.3f}')
    print(f'Micro Precision: {_micro_precision(y_true, y_pred):.3f}')
    print(f'Weighted Precision: {_weighted_precision(y_true, y_pred):.3f}')

    print(f'Macro Recall: {_macro_recall(y_true, y_pred):.3f}')
    print(f'Micro Recall: {_micro_recall(y_true, y_pred):.3f}')
    print(f'Weighted Recall: {_weighted_recall(y_true, y_pred):.3f}')

    print(f'Macro F1-score: {_macro_f1(y_true, y_pred):.3f}')
    print(f'Micro F1-score: {_micro_f1(y_true, y_pred):.3f}')
    print(f'Weighted F1-score: {_weighted_f1(y_true, y_pred):.3f}')

    _plot_confusion_matrix(y_true, y_pred)


def _macro_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    class_count = len(np.unique(y_true))
    macro_precision = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        macro_precision += _precision_postive_class(temp_true, temp_pred)
    return macro_precision / class_count


def _micro_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    class_count = len(np.unique(y_true))
    tp_count = 0
    fp_count = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        tp_count += _tp_binary(temp_true, temp_pred)
        fp_count += _fp_binary(temp_true, temp_pred)
    micro_precision = tp_count / (tp_count + fp_count)
    return micro_precision

def _weighted_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    class_count = len(np.unique(y_true))
    weighted_precision = 0
    class_fr = Counter(y_true)
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        weighted_precision += _precision_postive_class(temp_true, temp_pred) * class_fr[class_]
    return weighted_precision / len(y_true)


def _macro_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    class_count = len(np.unique(y_true))
    macro_recall = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        macro_recall += _recall_postive_class(temp_true, temp_pred)
    return macro_recall / class_count


def _micro_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.recall_score(y_true=y_true, y_pred=y_pred, average='micro')
    class_count = len(np.unique(y_true))
    tp_count = 0
    fn_count = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        tp_count += _tp_binary(temp_true, temp_pred)
        fn_count += _fn_binary(temp_true, temp_pred)
    micro_recall = tp_count / (tp_count + fn_count)
    return micro_recall


def _weighted_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    class_count = len(np.unique(y_true))
    weighted_recall = 0
    class_fr = Counter(y_true)
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]

        weighted_recall += _recall_postive_class(temp_true, temp_pred) * class_fr[class_]
    return weighted_recall / len(y_true)


def _macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')

    class_count = len(np.unique(y_true))
    macro_f1_score = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]
        precision = _precision_postive_class(temp_true, temp_pred)
        recall = _recall_postive_class(temp_true, temp_pred)
        if precision + recall == 0:
            pass
        else:
            macro_f1_score += (2 * (precision * recall) / (precision + recall))
    return macro_f1_score / class_count



def _micro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.f1_score(y_true=y_true, y_pred=y_pred, average='micro')

    precision = _micro_precision(y_true, y_pred)
    recall = _micro_recall(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall)


def _weighted_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    class_count = len(np.unique(y_true))
    class_fr = Counter(y_true)
    weighted_f1_score = 0
    for class_ in range(class_count):
        temp_true = [1 if class_ == class_pred else 0 for class_pred in y_true]
        temp_pred = [1 if class_ == class_pred else 0 for class_pred in y_pred]
        precision = _precision_postive_class(temp_true, temp_pred)
        recall = _recall_postive_class(temp_true, temp_pred)
        if precision + recall == 0:
            pass
        else:
            weighted_f1_score += (2 * (precision * recall) / (precision + recall)) * class_fr[class_]
    return weighted_f1_score / len(y_true)


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> None:
    cm = metrics.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0,
    as_cmap=True)
    sns.heatmap(cm, annot=True, cmap=cmap, cbar=False)
    plt.ylabel('Actual Labels', fontsize=20)
    plt.xlabel('Predicted Labels', fontsize=20)


if __name__ == '__main__':
    runner()
    plt.show()