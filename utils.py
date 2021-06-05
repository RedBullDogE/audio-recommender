import pandas as pd


def to_1D(arr_2d):
    """
    Функція-утиліта для перетворення двомірного масиву у одномірний
    """
    return pd.Series([x for _list in arr_2d for x in _list])


def normalize(df, cols=None):
    """
    Функція-утиліта для нормалізації даних стовпців вказаного датафрейму
    """
    result = df.copy()
    if cols is None:
        cols = df.columns

    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
