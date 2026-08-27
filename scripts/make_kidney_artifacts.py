"""DEPRECATED. Kidney artifacts are now produced by the leakage-safe trainer.

The previous version re-split the frame, loaded the persisted model, and scored
it on the same split to regenerate figures / minimal metadata -- re-using the
holdout, ignoring the id target-proxy, and one-hot encoding un-normalised
categorical tokens (dm_\tno, dm_ yes). All of that now lives in one place:

    python -m src.training.train_kidney

This shim simply forwards to it so any old muscle-memory entrypoint still works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.train_kidney import train_kidney  # noqa: E402

if __name__ == "__main__":
    train_kidney()
    print("Kidney artifacts regenerated via src.training.train_kidney")
