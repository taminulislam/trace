"""Step 1.7: Build master annotations.csv by merging all annotation sources.

Merges:
- frame_features.csv (frame_id, seq_id, paths, features, is_interpolated, excluded)
- class_mapping.csv (seq_id -> class_id, class_name, behavioural_label)
- split_train_val_test.csv (seq_id -> split)

Output: annotations/annotations.csv
"""

import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))


def main():
    features_df = pd.read_csv(ANNOTATIONS_DIR / "frame_features.csv")
    class_map_df = pd.read_csv(ANNOTATIONS_DIR / "class_mapping.csv")
    splits_df = pd.read_csv(ANNOTATIONS_DIR / "split_train_val_test.csv")

    features_df["seq_id"] = features_df["seq_id"].astype(str).str.zfill(4)
    class_map_df["seq_id"] = class_map_df["seq_id"].astype(str).str.zfill(4)
    splits_df["seq_id"] = splits_df["seq_id"].astype(str).str.zfill(4)

    merged = features_df.merge(
        class_map_df[["seq_id", "class_id", "class_name", "behavioural_label"]],
        on="seq_id",
        how="left",
    )
    merged = merged.merge(splits_df, on="seq_id", how="left")

    if "excluded" not in merged.columns:
        merged["excluded"] = False
    merged["excluded"] = merged["excluded"].fillna(False)

    columns = [
        "frame_id", "seq_id", "class_id", "behavioural_label",
        "image_path", "mask_path", "overlay_path",
        "gas_area_pct", "gas_centroid_x", "gas_centroid_y",
        "gas_dispersion", "gas_connected_components",
        "gas_intensity_mean", "gas_intensity_max",
        "is_interpolated", "excluded", "split",
    ]
    merged = merged[columns]

    out_path = ANNOTATIONS_DIR / "annotations.csv"
    merged.to_csv(out_path, index=False)

    print("Master Annotation CSV - Step 1.7")
    print("=" * 50)
    print(f"Total rows: {len(merged)}")
    print(f"Columns: {list(merged.columns)}")
    print(f"\nPer-class counts (all frames):")
    print(merged.groupby("class_id").size().to_string())
    print(f"\nPer-class counts (non-excluded):")
    active = merged[~merged["excluded"]]
    print(active.groupby("class_id").size().to_string())
    print(f"\nPer-split counts (non-excluded):")
    print(active.groupby("split").size().to_string())
    print(f"\nInterpolated frames: {merged['is_interpolated'].sum()}")
    print(f"Excluded frames: {merged['excluded'].sum()}")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
