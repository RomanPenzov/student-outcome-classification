"""Проверь сохранение, split и повторяемость на искусственном CSV."""

import json

import numpy as np
import pandas as pd
import pytest

from student_outcome.config import Config, read_config
from student_outcome.experiment import run_experiment
from student_outcome.models import make_models


def test_experiment(frame, tmp_path):
    path = tmp_path / "data.csv"
    frame.to_csv(path, sep=";", index=False)
    models = {k: v for k, v in make_models().items() if k in {"Dummy", "DecisionTree"}}
    tables = []
    for i in range(2):
        config = Config(path, tmp_path / str(i), folds=3)
        tables.append(run_experiment(config, models))
    cols = [c for c in tables[0] if c != "cv_seconds"]
    pd.testing.assert_frame_equal(tables[0][cols], tables[1][cols])
    split = json.loads((tmp_path / "0/split.json").read_text())
    assert set(split["train"]).isdisjoint(split["test"])
    validation = [v for f in split["folds"] for v in f["validation"]]
    assert sorted(validation) == sorted(split["train"])
    for fold in split["folds"]:
        assert set(fold["train"]).isdisjoint(fold["validation"])
        assert set(fold["validation"]).isdisjoint(split["test"])
    result = json.loads((tmp_path / "0/test_metrics.json").read_text())
    assert np.sum(result["confusion_matrix"]) == len(split["test"])
    assert (tmp_path / "0/model.joblib").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(Config(path, tmp_path / "0", folds=3), models)


def test_config_relative_paths(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('data_path = "data.csv"\noutput_dir = "out"\n')
    assert read_config(path).data_path == tmp_path / "data.csv"


@pytest.mark.parametrize("folds,test_size", [(1, 0.2), (5, 0), (5, 1)])
def test_invalid_config(tmp_path, folds, test_size):
    with pytest.raises(ValueError):
        Config(tmp_path / "data.csv", tmp_path / "out", folds=folds, test_size=test_size)
