"""Step 2.2: Temporal clip extraction.

Extracts clips of T=16 frames with stride=8 (50% overlap) from each sequence.
Stores clips as lists of frame paths (no file copying).
Outputs: annotations/clips.csv
"""

import os
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))
DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", _ROOT / "dataset"))

CLIP_LENGTH = 16
STRIDE = 8


def extract_clips_for_sequence(seq_frames: pd.DataFrame, clip_length: int,
                                stride: int) -> list:
    """Extract overlapping clips from a sorted sequence of frames.

    Args:
        seq_frames: DataFrame rows for one sequence, sorted by frame_id
        clip_length: number of frames per clip
        stride: step between clip start positions

    Returns:
        list of dicts, each containing clip metadata and frame paths
    """
    n_frames = len(seq_frames)
    if n_frames < clip_length:
        return []

    clips = []
    seq_id = seq_frames.iloc[0]["seq_id"]
    class_id = seq_frames.iloc[0]["class_id"]
    split = seq_frames.iloc[0]["split"]

    for start in range(0, n_frames - clip_length + 1, stride):
        clip_frames = seq_frames.iloc[start:start + clip_length]
        image_paths = clip_frames["image_path"].tolist()
        mask_paths = clip_frames["mask_path"].tolist()
        overlay_paths = clip_frames["overlay_path"].tolist()
        frame_ids = clip_frames["frame_id"].tolist()

        clips.append({
            "clip_id": f"{seq_id}_clip_{start:04d}",
            "seq_id": seq_id,
            "class_id": class_id,
            "split": split,
            "start_idx": start,
            "end_idx": start + clip_length - 1,
            "n_frames": clip_length,
            "frame_ids": "|".join(frame_ids),
            "image_paths": "|".join(image_paths),
            "mask_paths": "|".join(mask_paths),
            "overlay_paths": "|".join(overlay_paths),
        })

    return clips


def main():
    ann_df = pd.read_csv(ANNOTATIONS_DIR / "annotations.csv")
    active = ann_df[~ann_df["excluded"]].copy()

    active["seq_id"] = active["seq_id"].astype(str).str.zfill(4)
    active = active.sort_values(["seq_id", "frame_id"]).reset_index(drop=True)

    all_clips = []
    print("Temporal Clip Extraction - Step 2.2")
    print(f"Clip length: {CLIP_LENGTH}, Stride: {STRIDE}")
    print("=" * 50)

    for seq_id, seq_group in active.groupby("seq_id"):
        clips = extract_clips_for_sequence(seq_group, CLIP_LENGTH, STRIDE)
        all_clips.extend(clips)
        print(f"  SEQ_{seq_id}: {len(seq_group)} frames -> {len(clips)} clips")

    clips_df = pd.DataFrame(all_clips)
    out_path = ANNOTATIONS_DIR / "clips.csv"
    clips_df.to_csv(out_path, index=False)

    print(f"\nTotal clips: {len(clips_df)}")
    print(f"\nPer-split:")
    print(clips_df.groupby("split").size().to_string())
    print(f"\nPer-class:")
    print(clips_df.groupby("class_id").size().to_string())
    print(f"\nPer-class-split:")
    print(clips_df.groupby(["split", "class_id"]).size().to_string())
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
