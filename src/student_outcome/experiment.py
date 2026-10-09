"""Проведи единый benchmark и оцени выбранную модель на прежнем holdout."""

import platform
import warnings
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from student_outcome.artifacts import plot_confusion, save_json
from student_outcome.config import Config
from student_outcome.data import load_data
from student_outcome.evaluation import evaluate
from student_outcome.features import ALLOWLIST, CLASSES, select_features
from student_outcome.models import make_models


def run_experiment(config: Config, models: dict | None = None) -> pd.DataFrame:
    """Сравни конфигурации по одинаковым folds, сохрани выбранный pipeline.

    Args:
        config: Пути и параметры split/CV.
        models: Необязательный набор pipeline для локальных проверок.

    Returns:
        Таблица CV, упорядоченная по macro F1; Dummy не участвует в выборе.

    Raises:
        FileExistsError: Если папка результатов уже существует.
        ValueError: Если данные не подходят для split/CV или нет реальной модели.
    """
    frame = load_data(config.data_path, config.download)
    x = select_features(frame)
    if frame.duplicated().any() or x.duplicated().any():
        raise ValueError("Duplicate profiles require a reviewed group split")
    y = frame["Target"].map(dict(zip(CLASSES, range(3), strict=True))).to_numpy()
    train, test = train_test_split(
        np.arange(len(frame)), stratify=y, test_size=config.test_size, random_state=config.seed
    )
    if np.bincount(y[train]).min() < config.folds:
        raise ValueError("Too few training examples per class for CV")
    folds = list(
        StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed).split(
            x.iloc[train], y[train]
        )
    )
    candidates = make_models(config.seed) if models is None else models
    if not set(candidates) - {"Dummy"}:
        raise ValueError("At least one real model is required")
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=False)
    save_json(
        out / "split.json",
        {
            "train": train.tolist(),
            "test": test.tolist(),
            "folds": [
                {"train": train[a].tolist(), "validation": train[b].tolist()} for a, b in folds
            ],
        },
    )
    rows, fold_rows = [], []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("error", category=ConvergenceWarning)
        for name, pipeline in candidates.items():
            print(f"CV: {name}", flush=True)
            oof = np.empty(len(train), dtype=int)
            start = perf_counter()
            scores = []
            for k, (a, b) in enumerate(folds):
                fitted = clone(pipeline).fit(x.iloc[train[a]], y[train[a]])
                pred = np.asarray(fitted.predict(x.iloc[train[b]])).reshape(-1).astype(int)
                oof[b] = pred
                metrics = evaluate(y[train[b]], pred)
                flat = {
                    key: metrics[key] for key in ["macro_f1", "weighted_f1", "balanced_accuracy"]
                }
                fold_rows.append({"model": name, "fold": k + 1, **flat})
                scores.append(flat)
            summary = pd.DataFrame(scores)
            oof_metrics = evaluate(y[train], oof)
            save_json(out / f"{name}_oof.json", oof_metrics)
            plot_confusion(out / f"{name}_oof.png", oof_metrics["confusion_matrix"], f"{name}: OOF")
            rows.append(
                {
                    "model": name,
                    "cv_macro_f1": summary.macro_f1.mean(),
                    "cv_macro_f1_std": summary.macro_f1.std(ddof=1),
                    "cv_weighted_f1": summary.weighted_f1.mean(),
                    "cv_balanced_accuracy": summary.balanced_accuracy.mean(),
                    "cv_seconds": perf_counter() - start,
                }
            )
        table = (
            pd.DataFrame(rows)
            .sort_values(["cv_macro_f1", "model"], ascending=[False, True])
            .reset_index(drop=True)
        )
        winner = str(table.loc[table.model.ne("Dummy"), "model"].iloc[0])
        save_json(out / "selection.json", {"winner": winner, "criterion": "mean CV macro F1"})
        fitted = clone(candidates[winner]).fit(x.iloc[train], y[train])
        pred = np.asarray(fitted.predict(x.iloc[test])).reshape(-1).astype(int)
        final = evaluate(y[test], pred)
    table.to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(out / "fold_metrics.csv", index=False)
    save_json(out / "test_metrics.json", final)
    plot_confusion(
        out / "test_confusion.png", final["confusion_matrix"], f"{winner}: historical holdout"
    )
    joblib.dump(
        {"pipeline": fitted, "classes": CLASSES, "features": ALLOWLIST}, out / "model.joblib"
    )
    save_json(
        out / "manifest.json",
        {
            "config": {
                k: str(v) if hasattr(v, "__fspath__") else v for k, v in asdict(config).items()
            },
            "data_sha256": sha256(config.data_path.read_bytes()).hexdigest(),
            "shape": list(frame.shape),
            "class_order": CLASSES,
            "features": ALLOWLIST,
            "winner": winner,
            "python": platform.python_version(),
            "versions": {
                pkg: version(pkg)
                for pkg in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "catboost",
                    "lightgbm",
                    "xgboost",
                    "joblib",
                ]
            },
            "parameters": {name: str(model.get_params()) for name, model in candidates.items()},
            "test_status": "Previously used in parts I-II; not a new independent test",
        },
    )
    return table
