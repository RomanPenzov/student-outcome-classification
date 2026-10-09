"""Проверь все семейства моделей и неизвестные категории."""

import numpy as np
import pytest

from student_outcome.features import CLASSES, select_features
from student_outcome.models import make_models


@pytest.mark.parametrize("name", list(make_models()))
def test_pipeline(frame, name):
    x = select_features(frame)
    y = frame.Target.map(dict(zip(CLASSES, range(3), strict=True))).to_numpy()
    model = make_models()[name]
    # Уменьшаем бюджет, не меняя семейство алгоритма.
    params = model.get_params()
    for key in ["model__n_estimators", "model__iterations"]:
        if key in params:
            model.set_params(**{key: 3})
    model.fit(x.iloc[:75], y[:75])
    unseen = x.iloc[75:].copy()
    unseen["Course"] = 999999
    unseen.loc[unseen.index[0], "Admission grade"] = np.nan
    pred = np.asarray(model.predict(unseen)).reshape(-1)
    assert pred.shape == (15,)
    assert set(pred) <= {0, 1, 2}
    if name == "CategoricalNB":
        encoded = model.named_steps["preprocessor"].transform(unseen)
        assert (encoded >= 0).all()
        assert (encoded % 1 == 0).all()
    assert (
        999999
        not in model.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .named_steps["encode"]
        .categories_[3]
    )
