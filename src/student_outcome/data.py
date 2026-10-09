"""Загрузи UCI CSV и проверь контракт обучающей таблицы."""

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from student_outcome.features import ALLOWLIST, CATEGORICAL, CLASSES

DATA_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"


def validate_schema(frame: pd.DataFrame) -> None:
    """Проверь обязательные поля, типы, цель и допустимые диапазоны.

    Args:
        frame: Непустая таблица после удаления пробелов в заголовках.

    Raises:
        ValueError: Если нарушена схема; NaN разрешены только в признаках.
    """
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("Empty table or duplicate columns")
    missing = set(ALLOWLIST + ["Target"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for name in ALLOWLIST:
        series = frame[name]
        if not is_numeric_dtype(series) or series.dtype == bool:
            raise ValueError(f"Numeric storage required: {name}")
        values = series.dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values: {name}")
        if name in CATEGORICAL and ((values < 0).any() or (values % 1 != 0).any()):
            raise ValueError(f"Nonnegative integer category required: {name}")
    for name in ["Admission grade", "Previous qualification (grade)"]:
        if not frame[name].dropna().between(0, 200).all():
            raise ValueError(f"Grade out of range: {name}")
    if not frame["Age at enrollment"].dropna().between(10, 100).all():
        raise ValueError("Age out of range")
    target = frame["Target"]
    if target.isna().any() or set(target.unique()) != set(CLASSES):
        raise ValueError("Training target must contain exactly three UCI classes")


def load_data(path: Path, download: bool = False) -> pd.DataFrame:
    """Прочитай локальный CSV; скачай официальный архив только по запросу.

    Args:
        path: Путь к CSV с разделителем ';'.
        download: Разрешение загрузить отсутствующий файл из UCI.

    Returns:
        Проверенная таблица с нормализованными заголовками.

    Raises:
        FileNotFoundError: Если CSV отсутствует и загрузка выключена.
        ValueError: Если схема неверна.
    """
    if not path.exists() and download:
        with urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        with ZipFile(BytesIO(payload)) as archive:
            data = archive.read("data.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    frame = pd.read_csv(path, sep=";")
    frame.columns = frame.columns.str.strip()
    validate_schema(frame)
    return frame
