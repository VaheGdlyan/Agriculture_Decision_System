"""
Block 4: Error-Driven EDA Dashboard
=====================================
Generates a 2x2 senior-level diagnostic dashboard from error_metadata.csv,
visualizing the model's phenomenological blind spots across four dimensions:
scale degradation, photometric fragility, the combined danger zone, and
shape deformation.

Output: outputs/error_analysis_dashboard.png
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for headless/server environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
CSV_PATH   = ROOT / "outputs" / "error_metadata.csv"
OUTPUT_PNG = ROOT / "outputs" / "error_analysis_dashboard.png"


def run_dashboard():
    logger.info("=" * 60)
    logger.info("  📊 Block 4: Error-Driven EDA Dashboard")
    logger.info("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────────────
    if not CSV_PATH.exists():
        logger.error(f"❌ error_metadata.csv not found: {CSV_PATH}")
        logger.error("   Run extract_metadata.py first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    logger.info(f"✅ Loaded {len(df)} rows from error_metadata.csv")

    # ── Global plotting style ──────────────────────────────────────────────────
    sns.set_theme(style="darkgrid", palette="viridis")

    # High-resolution figure with 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Master figure title
    fig.suptitle(
        "YOLOv8s Baseline Error Diagnostic: False Negative Distribution",
        fontsize=18,
        fontweight="bold",
        y=1.01,
        color="#E0E0E0",
    )

    # Shared dark grey background for the entire figure
    fig.patch.set_facecolor("#1C1C2E")
    for ax in axes.flat:
        ax.set_facecolor("#2A2A3E")
        ax.tick_params(colors="#B0B0C8")
        ax.xaxis.label.set_color("#B0B0C8")
        ax.yaxis.label.set_color("#B0B0C8")
        ax.title.set_color("#E8E8FF")
        for spine in ax.spines.values():
            spine.set_edgecolor("#44445A")

    # ── Subplot 1 (0,0): Scale Degradation ────────────────────────────────────
    ax1 = axes[0, 0]
    sns.histplot(
        data=df,
        x="Area",
        kde=True,
        color="#7B61FF",
        ax=ax1,
        bins=30,
    )
    ax1.set_title("Scale Degradation (Area)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Bounding Box Area (px²)")
    ax1.set_ylabel("Count")
    # Annotate the sub-ERF danger zone
    ax1.axvline(x=1024, color="#FF6B6B", linestyle="--", linewidth=1.5, label="32×32 px (ERF limit)")
    ax1.legend(fontsize=9, labelcolor="#E0E0E0", facecolor="#1C1C2E", edgecolor="#44445A")

    # ── Subplot 2 (0,1): Photometric Fragility ────────────────────────────────
    ax2 = axes[0, 1]
    sns.histplot(
        data=df,
        x="Mean_Luminance",
        kde=True,
        color="#00C9A7",
        ax=ax2,
        bins=30,
    )
    ax2.set_title("Photometric Fragility (Luminance)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Mean Pixel Luminance (0–255)")
    ax2.set_ylabel("Count")
    # Annotate the darkness threshold
    ax2.axvline(x=80, color="#FF6B6B", linestyle="--", linewidth=1.5, label="Low-light (<80)")
    ax2.legend(fontsize=9, labelcolor="#E0E0E0", facecolor="#1C1C2E", edgecolor="#44445A")

    # ── Subplot 3 (1,0): The Danger Zone ──────────────────────────────────────
    ax3 = axes[1, 0]
    sns.kdeplot(
        data=df,
        x="Area",
        y="Mean_Luminance",
        fill=True,
        cmap="mako",
        ax=ax3,
        levels=12,
        thresh=0.02,
    )
    ax3.set_title("The Danger Zone: Area vs Luminance", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("Bounding Box Area (px²)")
    ax3.set_ylabel("Mean Luminance")
    # Annotate quadrant labels
    ax3.text(
        0.03, 0.93, "⚠ CRITICAL\n(Small + Dark)",
        transform=ax3.transAxes, fontsize=8.5,
        color="#FF6B6B", fontweight="bold", va="top",
    )

    # ── Subplot 4 (1,1): Shape Deformation ────────────────────────────────────
    ax4 = axes[1, 1]
    sns.histplot(
        data=df,
        x="Aspect_Ratio",
        kde=True,
        color="#FFB347",
        ax=ax4,
        bins=30,
    )
    ax4.set_title("Shape Deformation (Aspect Ratio)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Aspect Ratio (W / H)")
    ax4.set_ylabel("Count")
    # Mark the "square" reference line
    ax4.axvline(x=1.0, color="#C0C0E0", linestyle="--", linewidth=1.5, label="Square (1.0)")
    ax4.legend(fontsize=9, labelcolor="#E0E0E0", facecolor="#1C1C2E", edgecolor="#44445A")

    # ── Layout & Save ──────────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(OUTPUT_PNG),
        dpi=180,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    logger.info("=" * 60)
    logger.info(f"✅ Dashboard saved at: {OUTPUT_PNG}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_dashboard()
