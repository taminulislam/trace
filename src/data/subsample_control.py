"""Step 1.5: Control class subsampling.

Randomly mark ~296 frames from SEQ_0499 and SEQ_0501 (largest Control seqs)
as excluded=True to bring Control class from ~2,296 to ~2,000 frames.
No file deletion -- exclusion tracked in annotations only.
Seed=42 for reproducibility.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", _ROOT / "dataset"))
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))

CLASS_MAPPING = {
    "0483": 0, "0484": 0, "0486": 0, "0488": 0, "0490": 0,
    "0491": 0, "0492": 0, "0493": 0, "0495": 0, "0496": 0,
    "0497": 0, "0498": 0, "0510": 0,
    "0499": 1, "0500": 1, "0501": 1, "0502": 1, "0503": 1, "0504": 1,
    "0505": 2, "0506": 2, "0507": 2, "0508": 2, "0509": 2,
}

CONTROL_TARGET = 2000
SEQS_TO_SUBSAMPLE = ["0499", "0501"]
SEED = 42


def main():
    df = pd.read_csv(ANNOTATIONS_DIR / "frame_features.csv")

    control_seqs = [s for s, c in CLASS_MAPPING.items() if c == 1]
    control_frames = df[df["seq_id"].astype(str).str.zfill(4).isin(control_seqs)]
    current_count = len(control_frames)
    n_to_exclude = current_count - CONTROL_TARGET

    print(f"Control class: {current_count} frames total")
    print(f"Target: {CONTROL_TARGET} frames")
    print(f"Need to exclude: {n_to_exclude} frames")

    if n_to_exclude <= 0:
        print("No subsampling needed")
        df["excluded"] = False
        df.to_csv(ANNOTATIONS_DIR / "frame_features.csv", index=False)
        return

    subsample_pool = df[
        df["seq_id"].astype(str).str.zfill(4).isin(SEQS_TO_SUBSAMPLE)
    ]
    print(f"Subsampling pool: {len(subsample_pool)} frames from {SEQS_TO_SUBSAMPLE}")

    rng = np.random.RandomState(SEED)
    exclude_indices = rng.choice(
        subsample_pool.index,
        size=min(n_to_exclude, len(subsample_pool)),
        replace=False,
    )

    df["excluded"] = False
    df.loc[exclude_indices, "excluded"] = True

    n_excluded = df["excluded"].sum()
    remaining_control = len(control_frames) - n_excluded
    print(f"Excluded {n_excluded} frames")
    print(f"Remaining Control frames: {remaining_control}")

    for seq_id in SEQS_TO_SUBSAMPLE:
        seq_mask = df["seq_id"].astype(str).str.zfill(4) == seq_id
        seq_excluded = df[seq_mask & df["excluded"]].shape[0]
        seq_total = df[seq_mask].shape[0]
        print(f"  SEQ_{seq_id}: excluded {seq_excluded}/{seq_total}")

    df.to_csv(ANNOTATIONS_DIR / "frame_features.csv", index=False)
    print(f"\nUpdated {ANNOTATIONS_DIR / 'frame_features.csv'}")


if __name__ == "__main__":
    main()
