"""Определи фиксированные конфигурации общего benchmark."""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from student_outcome.preprocessing import make_preprocessor


def make_models(seed: int = 42) -> dict[str, Pipeline]:
    """Создай одиннадцать новых pipeline без разделяемого обученного состояния.

    Args:
        seed: Seed алгоритмов; потоки библиотечных бустингов ограничены одним.

    Returns:
        Словарь: имя модели → необученный pipeline.
    """
    estimators = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            C=1, max_iter=3000, class_weight="balanced", random_state=seed
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=seed
        ),
        "SVM_linear": SVC(kernel="linear", C=1, class_weight="balanced", random_state=seed),
        "SVM_rbf": SVC(
            kernel="rbf", C=1, gamma="scale", class_weight="balanced", random_state=seed
        ),
        "CategoricalNB": CategoricalNB(alpha=1, min_categories=5),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            num_leaves=15,
            learning_rate=0.05,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
            n_jobs=1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            thread_count=1,
        ),
    }
    return {
        name: Pipeline(
            [("preprocessor", make_preprocessor(name == "CategoricalNB")), ("model", model)]
        )
        for name, model in estimators.items()
    }
