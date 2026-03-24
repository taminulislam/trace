"""Step 1.6: Video-level train/val/test split.

Strategy: Stratified by class at SEQ level (NEVER frame level).
- Train: HF=9, Control=4, LF=3  (16 SEQs)
- Val:   HF=2, Control=1, LF=1  (4 SEQs)
- Test:  HF=2, Control=1, LF=1  (4 SEQs)

Seed=42 for reproducibility.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))
SEED = 42

HF_SEQS = ["0483", "0484", "0486", "0488", "0490", "0491", "0492", "0493",
            "0495", "0496", "0497", "0498", "0510"]
CONTROL_SEQS = ["0499", "0500", "0501", "0502", "0503", "0504"]
LF_SEQS = ["0505", "0506", "0507", "0508", "0509"]

SPLIT_CONFIG = {
    "HF": {"seqs": HF_SEQS, "train": 11, "val": 1, "test": 1},
    "Control": {"seqs": CONTROL_SEQS, "train": 4, "val": 1, "test": 1},
    "LF": {"seqs": LF_SEQS, "train": 3, "val": 1, "test": 1},
}


def main():
    rng = np.random.RandomState(SEED)
    rows = []

    print("Video-Level Train/Val/Test Split - Step 1.6")
    print("=" * 50)

    for class_name, config in SPLIT_CONFIG.items():
        seqs = list(config["seqs"])
        rng.shuffle(seqs)

        n_train = config["train"]
        n_val = config["val"]

        train_seqs = seqs[:n_train]
        val_seqs = seqs[n_train:n_train + n_val]
        test_seqs = seqs[n_train + n_val:]

        print(f"\n{class_name}:")
        print(f"  Train ({len(train_seqs)}): {train_seqs}")
        print(f"  Val   ({len(val_seqs)}): {val_seqs}")
        print(f"  Test  ({len(test_seqs)}): {test_seqs}")

        for s in train_seqs:
            rows.append({"seq_id": s, "split": "train"})
        for s in val_seqs:
            rows.append({"seq_id": s, "split": "val"})
        for s in test_seqs:
            rows.append({"seq_id": s, "split": "test"})

    df = pd.DataFrame(rows)
    out_path = ANNOTATIONS_DIR / "split_train_val_test.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSplit CSV written to {out_path}")
    print(f"Total: {len(df)} sequences")
    print(f"  Train: {len(df[df['split'] == 'train'])}")
    print(f"  Val:   {len(df[df['split'] == 'val'])}")
    print(f"  Test:  {len(df[df['split'] == 'test'])}")


if __name__ == "__main__":
    main()
