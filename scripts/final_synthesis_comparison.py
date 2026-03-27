"""
Block 6: Global Error Synthesis Dashboard
==========================================
Overlays the statistical DNA of False Negatives (Misses) against False Positives
(Hallucinations) in a research-paper quality 1×3 figure.

Outputs: outputs/pro_error_synthesis.png
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
FN_CSV      = ROOT / "outputs" / "error_metadata.csv"
FP_JSON     = ROOT / "outputs" / "fp_coordinates.json"
LABELS_DIR  = ROOT / "data" / "processed" / "yolo" / "labels" / "val"
IMAGES_DIR  = ROOT / "data" / "processed" / "yolo" / "images" / "val"
OUTPUT_PNG  = ROOT / "outputs" / "pro_error_synthesis.png"

IMG_SIZE    = 1024
PADDING     = 20

# Brand colors
C_FN = "#00FFFF"   # Neon Cyan  — False Negatives (Misses)
C_FP = "#FF004D"   # Crimson    — False Positives (Hallucinations)


# ─── 1. Load FN data from CSV ─────────────────────────────────────────────────
def load_fn_df() -> pd.DataFrame:
    if not FN_CSV.exists():
        logger.error(f"❌ error_metadata.csv not found: {FN_CSV}")
        sys.exit(1)
    df = pd.read_csv(FN_CSV)[["Area", "Mean_Luminance"]].copy()
    df["Error_Type"] = "FN (Miss)"
    return df


# ─── 2. Extract FP Area + Luminance from fp_coordinates.json + gallery crops ─
def load_fp_df() -> pd.DataFrame:
    if not FP_JSON.exists():
        logger.error(f"❌ fp_coordinates.json not found: {FP_JSON}")
        sys.exit(1)

    import cv2

    with open(FP_JSON) as f:
        fp_data = json.load(f)

    rows = []
    for img_path_str, records in fp_data.items():
        img_path = Path(img_path_str)
        if not img_path.exists():
            continue

        # Load grayscale once per image, reuse for all crops
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        img_h, img_w = gray.shape

        for rec in records:
            x1, y1, x2, y2 = rec["box"]

            # True geometric area from raw coordinates (no padding inflation)
            true_w = x2 - x1
            true_h = y2 - y1
            area   = true_w * true_h

            # Guard against zero/negative area anomalies before log scale
            if area <= 0:
                continue

            # Mean luminance from the padded crop (same methodology as FN)
            cx1 = int(max(0, x1 - PADDING))
            cy1 = int(max(0, y1 - PADDING))
            cx2 = int(min(img_w - 1, x2 + PADDING))
            cy2 = int(min(img_h - 1, y2 + PADDING))

            if cx2 <= cx1 or cy2 <= cy1:
                continue

            crop = gray[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            mean_lum = float(crop.mean())

            rows.append({"Area": area, "Mean_Luminance": mean_lum, "Error_Type": "FP (Hallucination)"})

    return pd.DataFrame(rows)


# ─── 3. Main dashboard renderer ───────────────────────────────────────────────
def run_synthesis():
    logger.info("=" * 60)
    logger.info("  🔬 Block 6: Global Error Synthesis Dashboard")
    logger.info("=" * 60)

    fn_df = load_fn_df()
    logger.info(f"✅ FN rows loaded: {len(fn_df)}")

    logger.info("⏳ Extracting FP features from gallery crops (this will take a moment)...")
    fp_df = load_fp_df()
    logger.info(f"✅ FP rows loaded: {len(fp_df)}")

    # Combined DataFrame
    df = pd.concat([fn_df, fp_df], ignore_index=True)
    logger.info(f"✅ Combined dataset: {len(df)} rows")

    # Safety: drop any residual zero/NaN area rows before log scale
    df = df[df["Area"] > 0].dropna(subset=["Area", "Mean_Luminance"])
    df["log_Area"] = np.log10(df["Area"])

    # ── Canvas ────────────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 7), facecolor="#0D0D1A")

    fig.suptitle(
        "YOLOv8s Global Error Synthesis: Localization vs. Detection Failures",
        fontsize=17, fontweight="bold", color="#F0F0FF", y=1.03,
    )

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Shared axis styling helper
    def style_ax(ax, title):
        ax.set_facecolor("#12121F")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#E8E8FF", pad=10)
        ax.tick_params(colors="#9090B0", labelsize=9)
        ax.xaxis.label.set_color("#9090B0")
        ax.yaxis.label.set_color("#9090B0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2A2A4A")

    # ── Plot 1: Intersection Map — KDE contour overlay ────────────────────────
    fn_data = df[df["Error_Type"] == "FN (Miss)"]
    fp_data = df[df["Error_Type"] == "FP (Hallucination)"]

    sns.kdeplot(
        data=fn_data, x="log_Area", y="Mean_Luminance",
        fill=False, levels=5, color=C_FN, linewidths=1.8, ax=ax1,
        label="FN (Miss)",
    )
    sns.kdeplot(
        data=fp_data, x="log_Area", y="Mean_Luminance",
        fill=False, levels=5, color=C_FP, linewidths=1.8, ax=ax1,
        label="FP (Hallucination)",
    )

    # X-axis: log10(Area) tick labels back to real px² values
    x_ticks = [np.log10(v) for v in [500, 1024, 2000, 5000, 10000, 25000]]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([f"{int(10**v):,}" for v in x_ticks], rotation=30, ha="right", fontsize=8)
    ax1.set_xlabel("Bounding Box Area (px², log scale)")
    ax1.set_ylabel("Mean Luminance")

    # ERF boundary reference line (log10(1024) ≈ 3.01)
    ax1.axvline(x=np.log10(1024), color="white", linestyle="--", linewidth=1.0, alpha=0.5, label="ERF limit (32×32)")
    ax1.legend(fontsize=8, facecolor="#0D0D1A", edgecolor="#2A2A4A", labelcolor="white")
    style_ax(ax1, "Plot 1 — The Intersection Map\n(Area vs Luminance Topology)")

    # ── Plot 2: Scale Asymmetry — Violin with log Y axis ─────────────────────
    df_scale = df[["Error_Type", "Area"]].copy()

    sns.violinplot(
        data=df_scale, x="Error_Type", y="Area",
        palette={"FN (Miss)": C_FN, "FP (Hallucination)": C_FP},
        ax=ax2, inner="quartile", linewidth=0.8, cut=0,
    )
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.axhline(y=1024, color="white", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.text(1.62, 1024 * 1.15, "ERF limit (32×32 px)", color="white", fontsize=7.5,
             va="bottom", ha="right", transform=ax2.get_yaxis_transform())
    ax2.set_xlabel("Error Type")
    ax2.set_ylabel("Bounding Box Area (px², log scale)")
    style_ax(ax2, "Plot 2 — Scale Asymmetry\n(Area Distribution by Error Type)")

    # ── Plot 3: Photometric Symmetry — Violin for luminance ──────────────────
    sns.violinplot(
        data=df[["Error_Type", "Mean_Luminance"]],
        x="Error_Type", y="Mean_Luminance",
        palette={"FN (Miss)": C_FN, "FP (Hallucination)": C_FP},
        ax=ax3, inner="quartile", linewidth=0.8, cut=0,
    )
    ax3.axhline(y=80, color="white", linestyle="--", linewidth=1.2, alpha=0.7)
    ax3.text(1.62, 82, "Low-light threshold (<80)", color="white", fontsize=7.5,
             va="bottom", ha="right", transform=ax3.get_yaxis_transform())
    ax3.set_xlabel("Error Type")
    ax3.set_ylabel("Mean Luminance (0–255)")
    style_ax(ax3, "Plot 3 — Photometric Symmetry\n(Luminance Distribution by Error Type)")

    # ── Save ──────────────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUTPUT_PNG), dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    logger.info(f"✅ Dashboard saved: {OUTPUT_PNG}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_synthesis()
