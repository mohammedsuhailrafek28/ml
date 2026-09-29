"""DEPRECATED. Liver artifacts are now produced by the leakage-safe trainer.

The previous version of this script re-split the *non-deduplicated* ILPD frame,
loaded the persisted model, and scored it on that same split to regenerate
figures / minimal metadata -- which re-used the hold-out for reporting and
ignored the 13 duplicate rows. All of that now lives in one place:

    python -m src.training.train_liver

This shim simply forwards to it so any old muscle-memory entrypoint still works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_liver import train_liver  # noqa: E402

if __name__ == "__main__":
    train_liver()
    print("Liver artifacts regenerated via src.training.train_liver")
