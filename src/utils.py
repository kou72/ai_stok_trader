"""
共通ユーティリティ関数
"""
import time
import numpy as np
from contextlib import contextmanager


def calculate_precision(predictions, targets):
    """
    モデルが1と予測したときの的中率（適合率）を計算

    Parameters:
    -----------
    predictions : ndarray
        予測確率
    targets : ndarray
        正解ラベル

    Returns:
    --------
    precision : float
        的中率
    """
    pred_binary = (predictions > 0.5).astype(int)
    predicted_positive = pred_binary == 1
    n_predicted_positive = np.sum(predicted_positive)

    if n_predicted_positive == 0:
        return 0.0

    true_positive = np.sum((pred_binary == 1) & (targets == 1))
    precision = true_positive / n_predicted_positive

    return float(precision)


def calculate_precision_with_counts(predictions, targets):
    """
    的中率と予測数・正解数を計算

    Parameters:
    -----------
    predictions : ndarray
        予測確率
    targets : ndarray
        正解ラベル

    Returns:
    --------
    dict : 的中率情報
        - predicted_count: 1と予測した数
        - correct_count: 正解数
        - precision: 的中率
        - precision_percent: 的中率（%表記文字列）
    """
    pred_binary = (predictions > 0.5).astype(int)
    predicted_positive = pred_binary == 1
    n_predicted_positive = int(np.sum(predicted_positive))

    if n_predicted_positive == 0:
        precision = 0.0
        correct_count = 0
    else:
        correct_count = int(np.sum((pred_binary == 1) & (targets == 1)))
        precision = correct_count / n_predicted_positive

    return {
        'predicted_count': n_predicted_positive,
        'correct_count': correct_count,
        'precision': precision,
        'precision_percent': f'{precision*100:.2f}%'
    }


def format_time(seconds):
    """
    秒数を時:分:秒形式に変換

    Parameters:
    -----------
    seconds : float
        秒数

    Returns:
    --------
    str : フォーマットされた時間文字列
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}時間{minutes}分{secs:.2f}秒"
    elif minutes > 0:
        return f"{minutes}分{secs:.2f}秒"
    else:
        return f"{secs:.2f}秒"


@contextmanager
def time_measure(time_log, key):
    """
    処理時間を計測するコンテキストマネージャ

    Parameters:
    -----------
    time_log : dict
        時間記録用の辞書
    key : str
        記録するキー名

    Usage:
    ------
    time_log = {}
    with time_measure(time_log, 'data_load'):
        # 計測したい処理
        pass
    print(time_log['data_load']['formatted'])
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        time_log[key] = {
            'seconds': elapsed,
            'formatted': format_time(elapsed)
        }
