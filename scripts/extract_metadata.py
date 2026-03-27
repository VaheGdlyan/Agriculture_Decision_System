"""
Block 3: Metadata Feature Engineering Engine
=============================================
Parses outputs/fn_coordinates.json and computes geometric properties
(True Width, Height, Area, Aspect Ratio) directly from the raw JSON
coordinates, then loads each corresponding cropped gallery image in
grayscale to compute Mean Luminance.

Output: outputs/error_metadata.csv
Columns: Image_Name, Box_Index, True_Width, True_Height, Area, Aspect_Ratio, Mean_Luminance
"""

import csv
import json
import logging
import sys
from pathlib import Path

import cv2

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
FN_JSON      = ROOT / "outputs" / "fn_coordinates.json"
GALLERY_DIR  = ROOT / "outputs" / "error_gallery"
OUTPUT_CSV   = ROOT / "outputs" / "error_metadata.csv"

CSV_COLUMNS = [
    "Image_Name",
    "Box_Index",
    "True_Width",
    "True_Height",
    "Area",
    "Aspect_Ratio",
    "Mean_Luminance",
]


def run_metadata_extraction():
    logger.info("=" * 60)
    logger.info("  📐 Block 3: Metadata Feature Engineering Engine")
    logger.info("=" * 60)

    # ── Infrastructure checks ──────────────────────────────────────────────────
    if not FN_JSON.exists():
        logger.error(f"❌ fn_coordinates.json not found: {FN_JSON}")
        logger.error("   Run extract_false_negatives.py first.")
        sys.exit(1)

    if not GALLERY_DIR.exists():
        logger.error(f"❌ Error gallery directory not found: {GALLERY_DIR}")
        logger.error("   Run generate_error_gallery.py first.")
        sys.exit(1)

    with open(FN_JSON, "r") as f:
        fn_data: dict[str, list[list[float]]] = json.load(f)

    total_boxes = sum(len(v) for v in fn_data.values())
    logger.info(f"✅ Loaded {len(fn_data)} images with {total_boxes} FN boxes")

    # ── Main feature extraction loop ───────────────────────────────────────────
    rows = []
    skipped = 0

    for img_path_str, boxes in fn_data.items():
        # HOW we link JSON coordinates to gallery files:
        #   The gallery filename was constructed in generate_error_gallery.py as:
        #       FN_{image_stem}_{box_index}.jpg
        #   We regenerate the same step from the image path key in the JSON and
        #   reconstruct the exact filename for each box_index. This is a
        #   deterministic bijection — no fuzzy matching or directory scanning
        #   needed. If the file is absent, we log a warning and continue.
        stem = Path(img_path_str).stem   # UUID portion only

        for box_idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            # WHY we compute geometry from JSON, NOT from the padded crop:
            #   The gallery images have 20px of context padding added to all sides.
            #   If we measured width from the crop image array, we would get:
            #       (x2-x1 + 40) instead of (x2-x1)
            #   That inflates every measurement by a fixed offset, which would
            #   corrupt Area (quadratically) and visually mislead any downstream
            #   histogram or scatter plot. The JSON coords reflect the true
            #   model-missed object boundary, which is the scientifically correct
            #   reference frame for our analysis.
            true_w  = x2 - x1
            true_h  = y2 - y1
            area    = true_w * true_h
            # Guard against degenerate zero-height boxes (should not exist, but safe)
            aspect  = round(true_w / true_h, 4) if true_h > 0 else 0.0

            # ── Locate corresponding gallery crop ──────────────────────────────
            crop_path = GALLERY_DIR / f"FN_{stem}_{box_idx}.jpg"

            if not crop_path.exists():
                logger.warning(f"   ⚠️  Gallery crop not found: {crop_path.name} — skipping")
                skipped += 1
                continue

            # ── Load in GRAYSCALE and compute mean luminance ───────────────────
            # cv2.IMREAD_GRAYSCALE collapses all channels into a single intensity
            # plane (0=black, 255=white). np.mean() over the entire crop gives the
            # average luminance, which is our photometric stress indicator.
            # A low value (< ~80) typically signals a shadowed or occluded region.
            gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)

            if gray is None:
                logger.warning(f"   ⚠️  cv2 failed to decode: {crop_path.name} — skipping")
                skipped += 1
                continue

            mean_lum = round(float(gray.mean()), 4)

            rows.append({
                "Image_Name"    : crop_path.name,
                "Box_Index"     : box_idx,
                "True_Width"    : round(true_w, 2),
                "True_Height"   : round(true_h, 2),
                "Area"          : round(area, 2),
                "Aspect_Ratio"  : aspect,
                "Mean_Luminance": mean_lum,
            })

    # ── Write CSV ──────────────────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ────────────────────────────────────────────────────────────────
    areas = [r["Area"] for r in rows]
    lums  = [r["Mean_Luminance"] for r in rows]

    logger.info("=" * 60)
    logger.info("  📊 METADATA EXTRACTION REPORT")
    logger.info("=" * 60)
    logger.info(f"   Rows written          : {len(rows)}")
    logger.info(f"   Rows skipped          : {skipped}")
    logger.info(f"   Area — min/mean/max   : {min(areas):.0f} / {sum(areas)/len(areas):.0f} / {max(areas):.0f} px²")
    logger.info(f"   Luminance — min/mean  : {min(lums):.1f} / {sum(lums)/len(lums):.1f}")
    logger.info(f"   Output CSV            : {OUTPUT_CSV}")
    logger.info("=" * 60)
    logger.info("✅ Success! error_metadata.csv is ready for statistical analysis.")


if __name__ == "__main__":
    run_metadata_extraction()
