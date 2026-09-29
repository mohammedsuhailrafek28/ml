"""DEPRECATED. Parkinson's artifacts are now produced by the subject-aware trainer.

The previous version re-split the frame, loaded the persisted model, and scored
it on the same split to regenerate figures / minimal metadata -- and its
metadata wrongly claimed no group split existed. All of that now lives in one
place, with GroupShuffleSplit + StratifiedGroupKFold throughout:

    python -m src.training.train_parkinsons

This shim simply forwards to it so any old muscle-memory entrypoint still works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_parkinsons import train_parkinsons  # noqa: E402

if __name__ == "__main__":
    train_parkinsons()
    print("Parkinson's artifacts regenerated via src.training.train_parkinsons")
