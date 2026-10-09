"""Прочитай воспроизводимую конфигурацию эксперимента."""

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Храни параметры запуска.

    Attributes:
        data_path: Локальный CSV.
        output_dir: Новая папка результатов; перезапись запрещена.
        seed: Seed split, CV и моделей.
        test_size: Доля контрольной выборки.
        folds: Число стратифицированных folds.
        download: Разрешена ли загрузка из UCI.
    """

    data_path: Path
    output_dir: Path
    seed: int = 42
    test_size: float = 0.2
    folds: int = 5
    download: bool = False

    def __post_init__(self) -> None:
        """Проверь диапазоны параметров до начала вычислений."""
        if not 0 < self.test_size < 1 or self.folds < 2:
            raise ValueError("Require 0 < test_size < 1 and folds >= 2")


def read_config(path: Path) -> Config:
    """Прочитай TOML с путями относительно самого файла конфигурации.

    Args:
        path: Путь к TOML.

    Returns:
        Проверенная конфигурация с абсолютными путями.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    for name in ["data_path", "output_dir"]:
        raw[name] = (path.resolve().parent / raw[name]).resolve()
    return Config(**raw)
