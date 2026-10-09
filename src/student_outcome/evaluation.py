"""Рассчитай метрики в фиксированном порядке классов."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from student_outcome.features import CLASSES


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Оцени целочисленные метки 0, 1, 2.

    Args:
        y_true: Истинные классы в порядке CLASSES.
        y_pred: Предсказанные классы в том же кодировании.

    Returns:
        Общие метрики, classification report и матрица: строки — истина.
    """
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
