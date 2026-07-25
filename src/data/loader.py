"""Dataset loading with explicit missing-file errors."""
from pathlib import Path
import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. See datasets/info for acquisition details."
        )
    frame = pd.read_csv(dataset_path, na_values=["?", "NA", "N/A", ""])
    if frame.empty:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    return frame

