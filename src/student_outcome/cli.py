"""Предоставь команду student-outcome для запуска эксперимента."""

import argparse
from pathlib import Path

from student_outcome.config import read_config
from student_outcome.experiment import run_experiment


def main() -> None:
    """Прочитай --config и запусти эксперимент с явными ошибками."""
    parser = argparse.ArgumentParser(description="Student outcome benchmark")
    parser.add_argument("--config", type=Path, default=Path("configs/default.toml"))
    args = parser.parse_args()
    print(run_experiment(read_config(args.config)).to_string(index=False))


if __name__ == "__main__":
    main()
