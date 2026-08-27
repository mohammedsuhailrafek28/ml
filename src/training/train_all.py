"""Train and persist the disease pipelines.

Every disease has a dedicated, leakage-safe trainer. This module is only a
dispatcher + CLI:

    python -m src.training.train_all                # retrain all five
    python -m src.training.train_all --disease heart  # retrain one

Each dedicated trainer owns loading, cleaning, splitting, CV, model selection,
evaluation, explainability, persistence and metadata. Do not add training logic
here.
"""
from __future__ import annotations

import argparse

from src.utils.config import DISEASES

# disease key -> "module:function" for its dedicated trainer.
TRAINERS: dict[str, tuple[str, str]] = {
    "liver": ("src.training.train_liver", "train_liver"),
    "heart": ("src.training.train_heart", "train_heart"),
    "diabetes": ("src.training.train_diabetes", "train_diabetes"),
    "kidney": ("src.training.train_kidney", "train_kidney"),
    "parkinsons": ("src.training.train_parkinsons", "train_parkinsons"),
}


def train_one(key: str) -> None:
    if key not in TRAINERS:
        raise KeyError(f"No dedicated trainer registered for {key!r}")
    module_name, func_name = TRAINERS[key]
    module = __import__(module_name, fromlist=[func_name])
    getattr(module, func_name)()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain disease pipelines.")
    parser.add_argument("--disease", choices=list(DISEASES))
    args = parser.parse_args()
    keys = [args.disease] if args.disease else list(DISEASES)
    for key in keys:
        try:
            train_one(key)
            print(f"trained {key}")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"SKIP {key}: {exc}")


if __name__ == "__main__":
    main()
