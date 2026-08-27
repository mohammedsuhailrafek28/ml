"""DEPRECATED. Diabetes artifacts are now produced by the leakage-safe trainer.

The previous version of this script re-split the frame, loaded the persisted
model, and scored it on that same split to regenerate figures / minimal
metadata -- and the zero-as-missing transform lived *outside* the persisted
pipeline (a train/serve mismatch). All of that now lives in one place:

    python -m src.training.train_diabetes

This shim simply forwards to it so any old muscle-memory entrypoint still works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_diabetes import train_diabetes  # noqa: E402

if __name__ == "__main__":
    train_diabetes()
    print("Diabetes artifacts regenerated via src.training.train_diabetes")
