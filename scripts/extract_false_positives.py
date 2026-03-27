"""
Block 5: Hallucination & Localization Diagnostic Engine
=========================================================
Inverts the FN diagnostic logic. Instead of iterating over GT boxes to find
misses, we iterate over PREDICTED boxes to find ones that don't match any GT:

  - Hallucination:      max(IoU[pred_j, :]) == 0    → predicts where nothing exists
  - Poor Localization:  0 < max(IoU[pred_j, :]) < 0.5 → finds something but draws a sloppy box

Outputs:
  outputs/fp_coordinates.json   — metadata with type + confidence per flawed prediction
  outputs/fp_gallery/           — cropped images of each flawed prediction
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import cv2
import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO

warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "outputs" / "yolov8_baseline_640" / "best.pt"
LABELS_DIR   = ROOT / "data" / "processed" / "yolo" / "labels" / "val"
IMAGES_DIR   = ROOT / "data" / "processed" / "yolo" / "images" / "val"
FP_JSON      = ROOT / "outputs" / "fp_coordinates.json"
GALLERY_DIR  = ROOT / "outputs" / "fp_gallery"

IMG_SIZE         = 1024
CONF_THRESHOLD   = 0.25
HALLUC_THRESHOLD = 0.0     # max IoU == 0  → hallucination
LOCALIZ_UPPER    = 0.5     # 0 < max IoU < 0.5 → poor localization
PADDING          = 20


# ─── Helpers ──────────────────────────────────────────────────────────────────
def parse_yolo_label(label_path: Path) -> torch.Tensor:
    """Parse YOLO .txt → pixel xyxy tensor (N, 4). Empty → (0,4)."""
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return torch.zeros((0, 4), dtype=torch.float32)
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        xc, yc, w, h = [float(p) * IMG_SIZE for p in parts[1:5]]
        boxes.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)


def find_image(stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def clamp(value: float, lo: float, hi: float) -> int:
    return int(max(lo, min(value, hi)))


def crop_and_save(image: cv2.typing.MatLike, box: list, out_path: Path, img_h: int, img_w: int):
    """Crop a prediction box with 20px padding, clamp to bounds, save as JPEG."""
    x1, y1, x2, y2 = box
    cx1 = clamp(x1 - PADDING, 0, img_w - 1)
    cy1 = clamp(y1 - PADDING, 0, img_h - 1)
    cx2 = clamp(x2 + PADDING, 0, img_w - 1)
    cy2 = clamp(y2 + PADDING, 0, img_h - 1)
    if cx2 <= cx1 or cy2 <= cy1:
        return False
    crop = image[cy1:cy2, cx1:cx2]
    return cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_fp_extraction():
    logger.info("=" * 60)
    logger.info("  👻 Block 5: Hallucination & Localization Diagnostic Engine")
    logger.info("=" * 60)

    if not WEIGHTS_PATH.exists():
        logger.error(f"❌ Weights not found: {WEIGHTS_PATH}")
        sys.exit(1)
    if not LABELS_DIR.exists():
        logger.error(f"❌ Labels dir not found: {LABELS_DIR}")
        sys.exit(1)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    logger.info(f"✅ Found {len(label_files)} val label files")
    logger.info("🧠 Loading YOLOv8 model...")
    model = YOLO(str(WEIGHTS_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Running on: {device.upper()}")

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    fp_records = {}         # { img_path_str: [ { box, conf, type } ] }
    total_halluc  = 0
    total_localiz = 0
    total_crops   = 0
    skipped       = 0

    logger.info("🚀 Starting False Positive / Hallucination extraction pass...")

    for label_path in tqdm(label_files, desc="Scanning val set", unit="img"):
        stem = label_path.stem

        # ── GT boxes ──────────────────────────────────────────────────────────
        gt_boxes = parse_yolo_label(label_path)   # (N_gt, 4) pixel xyxy

        # ── Source image ───────────────────────────────────────────────────────
        img_path = find_image(stem)
        if img_path is None:
            skipped += 1
            continue

        # ── Inference ──────────────────────────────────────────────────────────
        with torch.no_grad():
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESHOLD,
                verbose=False,
            )

        pred_boxes  = results[0].boxes.xyxy.cpu()     # (M, 4)
        pred_scores = results[0].boxes.conf.cpu()     # (M,)

        if pred_boxes.shape[0] == 0:
            continue   # No predictions → no FPs to log

        # ── Vectorized IoU — THIS IS THE FP AXIS ──────────────────────────────
        # Matrix shape: (N_gt, M_pred).
        # For FN  we took max over dim=1 (per GT row).
        # For FP  we take max over dim=0 (per PRED column).
        # This answers: "for each prediction, what is its best overlap with any GT?"
        if gt_boxes.shape[0] == 0:
            # No GT at all → every prediction is a hallucination
            max_iou_per_pred = torch.zeros(pred_boxes.shape[0])
        else:
            iou_matrix = box_iou(gt_boxes, pred_boxes)  # (N_gt, M_pred)
            max_iou_per_pred = iou_matrix.max(dim=0).values  # (M_pred,)

        # ── Categorize each prediction ─────────────────────────────────────────
        image_records = []

        for pred_idx in range(pred_boxes.shape[0]):
            max_iou = max_iou_per_pred[pred_idx].item()
            conf    = round(pred_scores[pred_idx].item(), 4)
            box     = pred_boxes[pred_idx].tolist()   # [x1, y1, x2, y2]

            if max_iou == HALLUC_THRESHOLD:
                error_type = "hallucination"
                total_halluc += 1
            elif max_iou < LOCALIZ_UPPER:
                error_type = "localization"
                total_localiz += 1
            else:
                continue   # Good prediction — skip

            image_records.append({
                "box"       : box,
                "conf"      : conf,
                "type"      : error_type,
                "max_iou"   : round(max_iou, 4),
            })

        if not image_records:
            continue

        # ── Load image and crop each flawed prediction ─────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        img_h, img_w = image.shape[:2]
        img_stem = img_path.stem

        for rec in image_records:
            conf_str  = f"{rec['conf']:.2f}".replace(".", "p")
            out_name  = f"FP_{rec['type']}_{conf_str}_{img_stem}.jpg"
            out_path  = GALLERY_DIR / out_name

            if crop_and_save(image, rec["box"], out_path, img_h, img_w):
                total_crops += 1
            else:
                logger.warning(f"   ⚠️  Degenerate crop skipped: {out_name}")

        fp_records[str(img_path)] = image_records

    # ── Write JSON ─────────────────────────────────────────────────────────────
    with open(FP_JSON, "w") as f:
        json.dump(fp_records, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    total_fp = total_halluc + total_localiz
    logger.info("=" * 60)
    logger.info("  📊 FP DIAGNOSTIC REPORT")
    logger.info("=" * 60)
    logger.info(f"   Total Flawed Predictions : {total_fp:,}")
    logger.info(f"     ├── Hallucinations     : {total_halluc:,}  (IoU == 0)")
    logger.info(f"     └── Poor Localizations : {total_localiz:,}  (0 < IoU < 0.5)")
    logger.info(f"   Gallery crops saved      : {total_crops:,}")
    logger.info(f"   Images skipped           : {skipped}")
    logger.info(f"   fp_coordinates.json      : {FP_JSON}")
    logger.info(f"   fp_gallery/              : {GALLERY_DIR}")
    logger.info("=" * 60)
    logger.info("✅ Success! FP diagnostic complete.")


if __name__ == "__main__":
    run_fp_extraction()
