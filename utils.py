import pandas as pd


def to_1D(arr_2d):
    """
    Функція-утиліта для перетворення двомірного масиву у одномірний
    """
    return pd.Series([x for _list in arr_2d for x in _list])
