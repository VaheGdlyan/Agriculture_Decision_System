"""
Block 2: Visual Extraction Engine
===================================
Loads outputs/fn_coordinates.json and crops each False Negative bounding box
from its source validation image. Adds 20px context padding (with boundary
clamping) and saves each crop to outputs/error_gallery/.

Naming convention: FN_{image_stem}_{box_index}.jpg
"""

import json
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# ─── Logger ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
FN_JSON     = ROOT / "outputs" / "fn_coordinates.json"
GALLERY_DIR = ROOT / "outputs" / "error_gallery"

# Context padding added to all four sides of every crop (pixels)
PADDING = 20


def clamp(value: float, lo: float, hi: float) -> int:
    """
    Clamp a float coordinate to [lo, hi] and return as int.

    HOW this prevents IndexError:
      OpenCV accesses image pixels via `image[y1:y2, x1:x2]`. If any
      coordinate is negative, Python's slice notation wraps around from the
      END of the array, silently producing a wrong or empty crop.
      If any coordinate exceeds the image dimension, the slice silently
      truncates — but in edge cases with uint8 math this can produce
      zero-size arrays which crash downstream imwrite().
      The explicit max(0, ...) and min(dim-1, ...) locks all coordinates
      firmly inside the valid index space BEFORE slicing, making crashes
      mathematically impossible.
    """
    return int(max(lo, min(value, hi)))


def run_gallery_extraction():
    logger.info("=" * 60)
    logger.info("  🖼️  Block 2: Visual Error Gallery Extraction")
    logger.info("=" * 60)

    # ── Infrastructure checks ──────────────────────────────────────────────────
    if not FN_JSON.exists():
        logger.error(f"❌ fn_coordinates.json not found at: {FN_JSON}")
        logger.error("   Run scripts/extract_false_negatives.py first.")
        sys.exit(1)

    with open(FN_JSON, "r") as f:
        fn_data: dict[str, list[list[float]]] = json.load(f)

    if not fn_data:
        logger.warning("⚠️  fn_coordinates.json is empty. Nothing to extract.")
        sys.exit(0)

    total_images = len(fn_data)
    total_boxes  = sum(len(v) for v in fn_data.values())
    logger.info(f"✅ Loaded {total_images} images with {total_boxes} FN boxes total")

    # ── Create output directory ────────────────────────────────────────────────
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {GALLERY_DIR}")

    # ── Main extraction loop ───────────────────────────────────────────────────
    saved   = 0
    skipped = 0

    for img_path_str, boxes in tqdm(fn_data.items(), desc="Cropping FN boxes", unit="img"):
        img_path = Path(img_path_str)

        if not img_path.exists():
            logger.warning(f"   ⚠️  Image not found, skipping: {img_path.name}")
            skipped += 1
            continue

        # WHY cv2 and how channels are handled:
        #   cv2.imread() loads images in BGR channel order (Blue-Green-Red),
        #   which is OpenCV's internal convention. Saving back with cv2.imwrite()
        #   expects the same BGR order — so NO conversion is needed at all.
        #   If we were displaying with matplotlib or saving to PIL, we would need
        #   cv2.cvtColor(img, cv2.COLOR_BGR2RGB). Since we are only reading →
        #   cropping → writing via cv2 throughout, the channel order stays
        #   consistent and there are zero blue-tint artifacts.
        image = cv2.imread(str(img_path))

        if image is None:
            logger.warning(f"   ⚠️  cv2 failed to decode: {img_path.name} — skipping")
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]   # (H, W, C)
        stem = img_path.stem              # UUID without extension

        for box_idx, box in enumerate(boxes):
            x1_raw, y1_raw, x2_raw, y2_raw = box

            # Apply 20px context padding on all sides, then clamp to image bounds
            # max_x = img_w - 1 to stay within valid column indices
            # max_y = img_h - 1 to stay within valid row indices
            x1 = clamp(x1_raw - PADDING, 0, img_w - 1)
            y1 = clamp(y1_raw - PADDING, 0, img_h - 1)
            x2 = clamp(x2_raw + PADDING, 0, img_w - 1)
            y2 = clamp(y2_raw + PADDING, 0, img_h - 1)

            # Guard against degenerate (zero-area) crops after clamping
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"   ⚠️  Degenerate crop after clamp ({stem}, box {box_idx}) — skipping")
                continue

            # Crop: OpenCV uses [row_start:row_end, col_start:col_end] aka [y:y, x:x]
            crop = image[y1:y2, x1:x2]

            # Build output filename
            out_name = f"FN_{stem}_{box_idx}.jpg"
            out_path = GALLERY_DIR / out_name

            # cv2.imwrite with quality=95 preserves enough detail for visual analysis
            success = cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if success:
                saved += 1
            else:
                logger.warning(f"   ⚠️  imwrite failed for: {out_name}")

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  📊 GALLERY EXTRACTION REPORT")
    logger.info("=" * 60)
    logger.info(f"   Crops saved successfully  : {saved}")
    logger.info(f"   Items skipped/failed      : {skipped}")
    logger.info(f"   Gallery location          : {GALLERY_DIR}")
    logger.info("=" * 60)

    if saved > 0:
        logger.info("✅ Success! Error gallery is ready for visual inspection.")
    else:
        logger.error("❌ No crops were saved. Check warnings above.")


if __name__ == "__main__":
    run_gallery_extraction()
