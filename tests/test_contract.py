"""Проверь схему и temporal feature policy."""

import numpy as np
import pytest

from student_outcome.data import load_data, validate_schema
from student_outcome.features import ALLOWLIST, UNCERTAIN, select_features


def test_valid_schema(frame):
    validate_schema(frame)


@pytest.mark.parametrize("column", ["Target", "Course", "Age at enrollment"])
def test_missing_column(frame, column):
    with pytest.raises(ValueError, match="Missing"):
        validate_schema(frame.drop(columns=column))


@pytest.mark.parametrize("value", ["wrong", np.inf, -1, 1.5])
def test_bad_category(frame, value):
    frame["Course"] = (
        frame["Course"].astype(object) if isinstance(value, str) else frame["Course"].astype(float)
    )
    frame.loc[0, "Course"] = value
    with pytest.raises(ValueError):
        validate_schema(frame)


@pytest.mark.parametrize("value", [None, "Unknown"])
def test_bad_target(frame, value):
    frame.loc[0, "Target"] = value
    with pytest.raises(ValueError, match="target"):
        validate_schema(frame)


def test_allowlist(frame):
    for name in UNCERTAIN:
        frame[name] = 99
    selected = select_features(frame)
    assert list(selected) == ALLOWLIST
    assert len(ALLOWLIST) == len(set(ALLOWLIST)) == 18
    assert "Target" not in selected
    assert not any("sem" in name for name in selected)
    assert set(UNCERTAIN).isdisjoint(selected)
    selected.iloc[0, 0] = 999
    assert frame.iloc[0, 0] != 999


def test_duplicate_columns(frame):
    frame.columns = ["Course"] * len(frame.columns)
    with pytest.raises(ValueError, match="duplicate"):
        validate_schema(frame)


def test_load_local(frame, tmp_path):
    path = tmp_path / "data.csv"
    frame.rename(columns={"Daytime/evening attendance": "Daytime/evening attendance\t"}).to_csv(
        path, sep=";", index=False
    )
    assert "Daytime/evening attendance" in load_data(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")
