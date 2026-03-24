"""Step 1.8: Generate behavioural description dataset for LLaVA fine-tuning.

20 template descriptions per class (60 total), programmatically varied
using gas feature values. Target: ~1,000 unique (frame_id, description) pairs.
Output: annotations/behaviour_descriptions.csv
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = Path(os.environ.get("ANNOTATIONS_DIR", _ROOT / "annotations"))
SEED = 42
TARGET_DESCRIPTIONS = 1000

HF_TEMPLATES = [
    "The bovine subject exhibits high metabolic activity with a gas emission area covering {area:.1f}% of the thermal field, centered at ({cx:.0f}, {cy:.0f}).",
    "Active rumination state detected. The thermal signature shows a {intensity}-intensity gas plume dispersed across {area:.1f}% of the frame.",
    "Elevated methane emission observed with {n_blobs} distinct gas region(s), characteristic of high forage dietary intake.",
    "The cow displays vigorous digestive processing. Gas dispersion index of {disp:.1f} indicates active eructation events.",
    "High forage feeding state confirmed: thermal overlay reveals {intensity} emission intensity with {area:.1f}% spatial coverage.",
    "Pronounced gas plume detected at coordinates ({cx:.0f}, {cy:.0f}) with a dispersion of {disp:.1f}, consistent with active rumination.",
    "The subject's metabolic profile indicates peak digestive activity. {n_blobs} separate emission zone(s) are visible across {area:.1f}% of the field.",
    "Thermal behavioural analysis reveals sustained high-emission state with mean gas intensity of {mean_int:.0f} and maximum of {max_int:.0f}.",
    "The animal is in an active rumination phase. Gas coverage of {area:.1f}% with {intensity} thermal signature suggests high forage consumption.",
    "High metabolic output detected: the gas plume extends over {area:.1f}% of the observation area, indicating robust digestive function.",
    "Elevated eructation activity observed. The thermal emission centroid at ({cx:.0f}, {cy:.0f}) reflects continuous gas release typical of high forage diet.",
    "The bovine demonstrates active gastric fermentation with {n_blobs} gas cluster(s) and a spatial dispersion of {disp:.1f}.",
    "Thermal imaging confirms high metabolic state: {area:.1f}% gas coverage with {intensity} emission intensity across the monitored region.",
    "The subject is actively ruminating with visible gas emission covering {area:.1f}% of the thermal field at {intensity} intensity.",
    "High forage intake evident: gas dispersion of {disp:.1f} and {n_blobs} emission zone(s) indicate sustained digestive processing.",
    "The cow exhibits peak metabolic behaviour with thermal gas signature centered at ({cx:.0f}, {cy:.0f}) and {area:.1f}% coverage.",
    "Active eructation state: the thermal plume shows {intensity} intensity with a spatial spread of {disp:.1f} units.",
    "Significant methane release detected. The high forage diet correlates with {area:.1f}% gas coverage and {n_blobs} distinct emission region(s).",
    "Behavioural state: active rumination. Gas emission parameters indicate {intensity} metabolic throughput with {area:.1f}% field coverage.",
    "The thermal profile shows vigorous digestive activity, with gas dispersion at {disp:.1f} and peak intensity of {max_int:.0f}.",
]

CONTROL_TEMPLATES = [
    "The bovine subject maintains a balanced homeostatic state with moderate gas emission covering {area:.1f}% of the thermal field.",
    "Normal digestive equilibrium observed. Gas coverage at {area:.1f}% with {intensity} intensity reflects a balanced 50:50 forage-concentrate diet.",
    "The cow displays stable metabolic behaviour with {n_blobs} gas region(s) and a dispersion of {disp:.1f}.",
    "Balanced homeostatic condition confirmed: the thermal signature shows moderate emission centered at ({cx:.0f}, {cy:.0f}).",
    "The subject's metabolic rate is within normal parameters. Gas area of {area:.1f}% and {intensity} emission intensity indicate dietary balance.",
    "Steady-state digestive processing observed. {n_blobs} emission zone(s) present with mean intensity of {mean_int:.0f}.",
    "The animal maintains thermal equilibrium with consistent gas output at {area:.1f}% coverage and {disp:.1f} dispersion.",
    "Balanced dietary intake reflected in moderate gas emission: {area:.1f}% spatial coverage with {intensity} thermal intensity.",
    "Homeostatic regulation observed. The gas plume at ({cx:.0f}, {cy:.0f}) indicates normal rumen function under balanced feeding.",
    "The bovine exhibits a stable metabolic profile. Gas dispersion of {disp:.1f} and {area:.1f}% coverage are consistent with control diet.",
    "Normal rumination behaviour detected with {n_blobs} gas cluster(s) at {intensity} intensity level.",
    "The thermal emission pattern indicates balanced digestive activity with gas centered at ({cx:.0f}, {cy:.0f}).",
    "Moderate methane output observed, consistent with a balanced forage-concentrate ratio. Coverage: {area:.1f}%, intensity: {intensity}.",
    "The subject shows typical homeostatic gas emission with {area:.1f}% coverage and {n_blobs} distinct region(s).",
    "Balanced metabolic state: thermal gas signature at {intensity} intensity with spatial dispersion of {disp:.1f}.",
    "The cow's digestive output is stable. Gas emission area of {area:.1f}% and mean intensity {mean_int:.0f} reflect dietary equilibrium.",
    "Normal behavioural state observed with moderate gas dispersion ({disp:.1f}) and {area:.1f}% thermal field coverage.",
    "The thermal profile indicates standard metabolic function under balanced dietary conditions with {n_blobs} emission zone(s).",
    "Controlled gas emission at ({cx:.0f}, {cy:.0f}) with {area:.1f}% coverage suggests balanced rumen activity.",
    "Homeostatic digestive behaviour: the thermal signature shows {intensity} gas intensity with peak value of {max_int:.0f}.",
]

LF_TEMPLATES = [
    "The bovine subject shows reduced gas emission at {area:.1f}% coverage, indicating digestive distress from low forage intake.",
    "Low emission state detected. The thermal signature reveals {intensity}-intensity gas output with only {area:.1f}% spatial coverage.",
    "Digestive distress evident: diminished gas emission with {n_blobs} small region(s) and dispersion of {disp:.1f}.",
    "The cow exhibits suppressed metabolic activity. Gas emission centered at ({cx:.0f}, {cy:.0f}) covers only {area:.1f}% of the thermal field.",
    "Low forage dietary stress reflected in reduced methane output: {area:.1f}% coverage with {intensity} emission intensity.",
    "The subject's thermal profile indicates compromised digestive function with diminished gas dispersion of {disp:.1f}.",
    "Reduced eructation activity observed. {n_blobs} faint gas region(s) at {intensity} intensity suggest low forage availability.",
    "Digestive distress state: the thermal emission is weak with mean intensity {mean_int:.0f} and maximum {max_int:.0f}.",
    "The animal displays signs of dietary insufficiency. Gas coverage of {area:.1f}% is significantly below normal levels.",
    "Low emission state confirmed: the gas plume at ({cx:.0f}, {cy:.0f}) shows minimal thermal signature with {area:.1f}% coverage.",
    "The bovine's metabolic output is suppressed. Gas dispersion of {disp:.1f} and {n_blobs} region(s) indicate low forage diet.",
    "Compromised rumination observed. The thermal gas signature at {intensity} intensity reflects reduced digestive throughput.",
    "The cow shows diminished methane release with only {area:.1f}% gas coverage, consistent with low forage feeding conditions.",
    "Low metabolic activity detected: gas emission parameters ({area:.1f}% area, {disp:.1f} dispersion) suggest dietary distress.",
    "The thermal profile reveals suppressed gas output at ({cx:.0f}, {cy:.0f}) with {n_blobs} barely visible emission zone(s).",
    "Digestive function appears compromised. Gas coverage at {area:.1f}% with {intensity} intensity indicates forage deficiency.",
    "Reduced behavioural activity: the thermal emission covers only {area:.1f}% of the field with peak intensity {max_int:.0f}.",
    "Low forage intake evident from the minimal gas emission pattern: {n_blobs} region(s) at {intensity} thermal intensity.",
    "The subject's gas emission is notably reduced. Spatial coverage of {area:.1f}% and dispersion {disp:.1f} indicate dietary stress.",
    "Thermal behavioural analysis indicates low emission state with {area:.1f}% coverage and mean intensity of {mean_int:.0f}.",
]

TEMPLATES = {0: HF_TEMPLATES, 1: CONTROL_TEMPLATES, 2: LF_TEMPLATES}


def intensity_label(mean_val):
    if mean_val > 160:
        return "high"
    elif mean_val > 100:
        return "moderate"
    else:
        return "low"


def fill_template(template, row):
    return template.format(
        area=row["gas_area_pct"],
        cx=row["gas_centroid_x"],
        cy=row["gas_centroid_y"],
        disp=row["gas_dispersion"],
        n_blobs=int(row["gas_connected_components"]),
        mean_int=row["gas_intensity_mean"],
        max_int=row["gas_intensity_max"],
        intensity=intensity_label(row["gas_intensity_mean"]),
    )


def main():
    rng = np.random.RandomState(SEED)
    ann_df = pd.read_csv(ANNOTATIONS_DIR / "annotations.csv")

    active = ann_df[~ann_df["excluded"]].copy()

    descriptions_per_class = TARGET_DESCRIPTIONS // 3
    remainder = TARGET_DESCRIPTIONS % 3

    rows = []
    for class_id in [0, 1, 2]:
        templates = TEMPLATES[class_id]
        class_frames = active[active["class_id"] == class_id]

        n_target = descriptions_per_class + (1 if class_id < remainder else 0)

        if len(class_frames) >= n_target:
            sampled = class_frames.sample(n=n_target, random_state=rng)
        else:
            sampled = class_frames.sample(n=n_target, replace=True, random_state=rng)

        for idx, (_, frame_row) in enumerate(sampled.iterrows()):
            template = templates[idx % len(templates)]
            description = fill_template(template, frame_row)
            rows.append({
                "frame_id": frame_row["frame_id"],
                "seq_id": frame_row["seq_id"],
                "class_id": class_id,
                "description": description,
            })

    desc_df = pd.DataFrame(rows)
    out_path = ANNOTATIONS_DIR / "behaviour_descriptions.csv"
    desc_df.to_csv(out_path, index=False)

    print("Behavioural Description Generation - Step 1.8")
    print("=" * 50)
    print(f"Total descriptions: {len(desc_df)}")
    print(f"Per-class counts:")
    print(desc_df.groupby("class_id").size().to_string())
    print(f"Unique frame_ids: {desc_df['frame_id'].nunique()}")
    print(f"\nSample descriptions:")
    for cid in [0, 1, 2]:
        sample = desc_df[desc_df["class_id"] == cid].iloc[0]
        print(f"\n  Class {cid}: {sample['description'][:120]}...")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
