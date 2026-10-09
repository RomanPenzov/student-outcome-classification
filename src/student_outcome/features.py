"""Зафиксируй политику признаков из части I."""

import pandas as pd

FEATURE_SPEC = [
    ("Marital status", "category", "Семейное положение из анкеты; код не является числом на шкале"),
    ("Application mode", "category", "Способ поступления известен при оформлении"),
    (
        "Application order",
        "category",
        "Приоритет программы в заявлении; интервалы не считаем равными",
    ),
    (
        "Course",
        "category",
        "Программа, на которую поступил студент; число является идентификатором",
    ),
    ("Daytime/evening attendance", "category", "Выбранная дневная или вечерняя форма"),
    (
        "Previous qualification",
        "category",
        "Предыдущее образование; коды не образуют числовую шкалу",
    ),
    ("Previous qualification (grade)", "numeric", "Оценка предыдущего образования, шкала 0–200"),
    ("Nacionality", "category", "Гражданство из анкеты; сохранено исходное написание UCI"),
    ("Mother's qualification", "category", "Образование матери из анкеты; номинальные коды"),
    ("Father's qualification", "category", "Образование отца из анкеты; номинальные коды"),
    ("Mother's occupation", "category", "Профессия матери, указанная при поступлении"),
    ("Father's occupation", "category", "Профессия отца, указанная при поступлении"),
    ("Admission grade", "numeric", "Вступительный балл, шкала 0–200"),
    ("Displaced", "category", "Анкетный статус переезда к месту обучения на дату поступления"),
    (
        "Educational special needs",
        "category",
        "Заявленные при поступлении особые образовательные потребности",
    ),
    ("Gender", "category", "Пол, записанный в анкете; коды UCI"),
    ("Age at enrollment", "numeric", "Возраст непосредственно при поступлении"),
    ("International", "category", "Международный статус при поступлении"),
]
ALLOWLIST = [row[0] for row in FEATURE_SPEC]
CATEGORICAL = [row[0] for row in FEATURE_SPEC if row[1] == "category"]
NUMERIC = [row[0] for row in FEATURE_SPEC if row[1] == "numeric"]
UNCERTAIN = [
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]
CLASSES = ["Dropout", "Enrolled", "Graduate"]


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Выбери только разрешённые признаки в фиксированном порядке.

    Args:
        frame: Таблица с анкетными полями; дополнительные столбцы игнорируются.

    Returns:
        Независимая копия 18 признаков.

    Raises:
        ValueError: Если отсутствует обязательный признак.
    """
    missing = set(ALLOWLIST) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    return frame.loc[:, ALLOWLIST].copy()
