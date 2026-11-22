"""Load labelled text from a folder-per-class corpus or from a CSV file."""
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def load_folder_corpus(root: str) -> Tuple[List[str], List[int]]:
    """Load a corpus laid out as root/neg/*.txt and root/pos/*.txt.

    Returns (texts, labels) where label 1 is positive and 0 is negative.
    """
    root_path = Path(root)
    texts, labels = [], []
    for label, name in ((0, "neg"), (1, "pos")):
        folder = root_path / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        for path in sorted(folder.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    return texts, labels


def load_csv(path: str, text_col: str = "text", label_col: str = "label") -> Tuple[List[str], List[int]]:
    """Load a CSV with a text column and a binary label column (0 or 1)."""
    frame = pd.read_csv(path)
    for col in (text_col, label_col):
        if col not in frame.columns:
            raise KeyError(f"Column '{col}' not found. Available: {list(frame.columns)}")
    frame = frame.dropna(subset=[text_col, label_col])
    return frame[text_col].astype(str).tolist(), frame[label_col].astype(int).tolist()
