"""Создай синтетические данные без сети."""

import numpy as np
import pandas as pd
import pytest

from student_outcome.features import CATEGORICAL, CLASSES


@pytest.fixture
def frame():
    rng = np.random.default_rng(12)
    n = 90
    data = {name: rng.integers(0, 4, n) for name in CATEGORICAL}
    data.update(
        {
            "Admission grade": rng.uniform(80, 190, n),
            "Previous qualification (grade)": rng.uniform(80, 190, n),
            "Age at enrollment": rng.integers(18, 55, n),
            "Target": np.tile(CLASSES, n // 3),
            "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
            "Curricular units 2nd sem (grade)": rng.uniform(0, 20, n),
        }
    )
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Network is forbidden in unit tests")

    monkeypatch.setattr("socket.socket.connect", fail)
