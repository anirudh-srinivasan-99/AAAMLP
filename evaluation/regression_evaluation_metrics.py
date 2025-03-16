import numpy as np
import sklearn.metrics as metrics


def runner():
    y_true = np.array([3.5, 2.8, 4.0, 5.6, 7.1])
    y_pred = np.array([3.2, 2.9, 4.1, 5.4, 6.8])

    print(f'MAE: {_mae(y_true, y_pred):.3f}')

    print(f'MSE: {_mse(y_true, y_pred):.3f}')
    print(f'RMSE: {_rmse(y_true, y_pred):.3f}')

    print(f'MSLE: {_msle(y_true, y_pred):.3f}')
    print(f'RMSLE: {_rmsle(y_true, y_pred):.3f}')

    print(f'MPE: {_mpe(y_true, y_pred):.3f}')
    print(f'MAPE: {_mape(y_true, y_pred):.3f}')

    print(f'R2-Score: {_r2(y_true, y_pred):.3f}')


def _mae(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.mean_absolute_error(y_true, y_pred)
    mae = []
    for yt, yp in zip(y_true, y_pred):
        mae.append(abs(yt - yp))
    return np.mean(mae)


def _rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return _mse(y_true, y_pred) ** 0.5


def _mse(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.mean_squared_error(y_true, y_pred)
    mse = []
    for yt, yp in zip(y_true, y_pred):
        mse.append((yt - yp) ** 2)
    return np.mean(mse)


def _rmsle(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return _msle(y_true, y_pred) ** 0.5


def _msle(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.mean_squared_log_error(y_true, y_pred)
    msle = []
    for yt, yp in zip(y_true, y_pred):
        msle.append((np.log(1 + yt) - np.log(1 + yp)) ** 2)
    return np.mean(msle)


def _mpe(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    pe = []
    for yt, yp in zip(y_true, y_pred):
        pe.append((yt - yp)/yt)
    return np.mean(pe) * 100


def _mape(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    ape = []
    for yt, yp in zip(y_true, y_pred):
        ape.append(abs(yt - yp)/yt)
    return np.mean(ape) * 100


def _r2(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    # return metrics.r2_score(y_true, y_pred)
    yt_avg = np.mean(y_true)
    numerator, denominator = 0, 0

    for yt, yp in zip(y_true, y_pred):
        numerator += (yp - yt) ** 2
        denominator += (yt - yt_avg) ** 2
    ratio = numerator / denominator

    return 1 - ratio


if __name__ == '__main__':
    runner()