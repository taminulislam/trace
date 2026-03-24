"""Generate comparison table from all experiment results.

Reads eval_results.json from each Exp*/ folder and produces
a combined markdown + CSV results table.

Usage:
    python scripts/compare_results.py
"""

import json
import os
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def load_experiment(exp_dir):
    """Load eval results from an experiment directory."""
    results_file = exp_dir / "eval_results" / "eval_results.json"
    if not results_file.exists():
        return None
    with open(results_file) as f:
        data = json.load(f)
    return data


def _seg(data, key, default=0):
    """Get a segmentation metric from either nested or flat JSON format."""
    nested = data.get("segmentation", {})
    if nested:
        return nested.get(key, default)
    # Flat format: keys like "seg/iou"
    return data.get(f"seg/{key}", default)


def _cls(data, key, default=0):
    """Get a classification metric from either nested or flat JSON format."""
    nested = data.get("classification", {})
    if nested:
        return nested.get(key, default)
    # Flat format: keys like "cls/accuracy"
    return data.get(f"cls/{key}", default)


def format_table():
    """Collect all experiment results and format as table."""
    experiments = sorted(PROJECT_ROOT.glob("Exp*_*"))
    if not experiments:
        print("No Exp*_ directories found!")
        return

    rows = []
    for exp_dir in experiments:
        data = load_experiment(exp_dir)
        if data is None:
            print(f"  ⚠ No results in {exp_dir.name}")
            continue

        model_name = exp_dir.name.split("_", 1)[1] if "_" in exp_dir.name else exp_dir.name

        row = {
            "Model": model_name,
            "Params (M)": f"{data.get('params_m', '?')}",
            # Segmentation
            "mIoU": f"{_seg(data, 'iou'):.4f}",
            "Dice": f"{_seg(data, 'dice'):.4f}",
            "Precision": f"{_seg(data, 'precision'):.4f}",
            "Recall": f"{_seg(data, 'recall'):.4f}",
            "BF1": f"{_seg(data, 'boundary_f1'):.4f}",
            "HD (px)": f"{_seg(data, 'hausdorff'):.2f}",
            "Centroid Err": f"{_seg(data, 'centroid_err'):.4f}",
            "Plume Area Rel Err": f"{_seg(data, 'plume_area_rel_err'):.4f}",
            # Classification
            "Acc": f"{_cls(data, 'accuracy'):.4f}",
            "BalAcc": f"{_cls(data, 'balanced_accuracy'):.4f}",
            "F1": f"{_cls(data, 'macro_f1'):.4f}",
            "AUC": f"{_cls(data, 'macro_auc_roc'):.4f}",
            "Kappa": f"{_cls(data, 'cohen_kappa'):.4f}",
        }
        rows.append(row)

    if not rows:
        print("No results found!")
        return

    # Print markdown table
    headers = list(rows[0].keys())
    print("\n" + "=" * 120)
    print("  MODEL COMPARISON TABLE")
    print("=" * 120)

    # Header
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    print(header_line)
    print(sep_line)

    # Rows
    for row in rows:
        line = "| " + " | ".join(row.values()) + " |"
        print(line)

    print()

    # Save to file
    output_md = PROJECT_ROOT / "comparison_results.md"
    with open(output_md, "w") as f:
        f.write("# TRACE — Model Comparison Results\n\n")
        f.write(f"Experiments: {len(rows)}\n\n")
        f.write("## Segmentation + Classification Metrics\n\n")
        f.write(header_line + "\n")
        f.write(sep_line + "\n")
        for row in rows:
            line = "| " + " | ".join(row.values()) + " |"
            f.write(line + "\n")
        f.write("\n\n---\n*Generated automatically from experiment results.*\n")
    print(f"📄 Table saved to: {output_md}")

    # Also save CSV
    output_csv = PROJECT_ROOT / "comparison_results.csv"
    with open(output_csv, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(row.values()) + "\n")
    print(f"📊 CSV saved to: {output_csv}")


if __name__ == "__main__":
    format_table()
