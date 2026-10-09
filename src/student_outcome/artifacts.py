"""Сохрани JSON и график без интерактивного интерфейса."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from student_outcome.features import CLASSES


def save_json(path: Path, value: dict) -> None:
    """Запиши читаемый JSON.

    Args:
        path: Путь внутри существующей папки.
        value: JSON-совместимый словарь.
    """
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_confusion(path: Path, matrix: list, title: str) -> None:
    """Сохрани матрицу ошибок в PNG.

    Args:
        path: Путь к PNG.
        matrix: Матрица 3×3, строки — истинные классы.
        title: Название оцениваемой модели.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=CLASSES).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
