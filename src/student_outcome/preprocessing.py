"""Собери преобразования, обучаемые отдельно внутри каждого fold."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from student_outcome.features import CATEGORICAL, NUMERIC


def shift_categories(values: np.ndarray) -> np.ndarray:
    """Зарезервируй нулевой код для неизвестной категории.

    Args:
        values: Коды OrdinalEncoder, где неизвестная категория равна -1.

    Returns:
        Неотрицательные целочисленные коды.
    """
    return (values + 1).astype(np.int64)


def make_preprocessor(naive_bayes: bool = False) -> ColumnTransformer:
    """Создай независимый preprocessing для смешанных признаков.

    Args:
        naive_bayes: Использовать категории и пять равномерных интервалов для NB.

    Returns:
        Трансформер, отбрасывающий все поля вне allowlist.
    """
    category: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="constant", fill_value=-2, keep_empty_features=True))
    ]
    numeric: list[tuple[str, BaseEstimator]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if naive_bayes:
        category += [
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("shift", FunctionTransformer(shift_categories)),
        ]
        numeric += [("bins", KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform"))]
    else:
        category += [("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        numeric += [("scale", StandardScaler())]
    return ColumnTransformer(
        [
            ("cat", Pipeline(category), CATEGORICAL),
            ("num", Pipeline(numeric), NUMERIC),
        ],
        remainder="drop",
    )
