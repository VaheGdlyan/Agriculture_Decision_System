"""
Block 1: Inference & IoU Diagnostic Engine
==========================================
Identifies False Negatives (FN) — ground-truth wheat head boxes the model
completely missed (max IoU with any prediction < 0.1) — across the entire
validation set. Outputs a JSON mapping image paths to their missed box coords.

Output: outputs/fn_coordinates.json
"""

import json
import logging
import sys
import warnings
from pathlib import Path

import torch
from torchvision.ops import box_iou
from tqdm import tqdm
from ultralytics import YOLO

# Suppress verbose ultralytics output during batch inference
warnings.filterwarnings("ignore")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Configure our own clean logger
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
OUTPUT_PATH  = ROOT / "outputs" / "fn_coordinates.json"

# YOLO normalizes against this image size during prepare_yolo.py
IMG_SIZE = 1024

# False Negative threshold: GT box is a total miss if max IoU with any pred < this
FN_IOU_THRESHOLD = 0.1

# Confidence threshold for predictions (same as generate_clean_preds.py)
CONF_THRESHOLD = 0.25


def parse_yolo_label(label_path: Path) -> torch.Tensor:
    """
    Parse a YOLO .txt label file and return boxes in pixel [x1, y1, x2, y2] format.
    
    YOLO stores: class xc_norm yc_norm w_norm h_norm
    We denormalize at IMG_SIZE and convert centre→corner format.
    
    Returns: FloatTensor of shape (N, 4), or empty (0, 4) for no boxes.
    """
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return torch.zeros((0, 4), dtype=torch.float32)

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Skip class index (parts[0]), we only have 1 class
        xc, yc, w, h = [float(p) * IMG_SIZE for p in parts[1:5]]
        x1 = xc - w / 2
        y1 = yc - h / 2
        x2 = xc + w / 2
        y2 = yc + h / 2
        boxes.append([x1, y1, x2, y2])

    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def find_image(stem: str) -> Path | None:
    """Locate the image file for a label stem (supports .png and .jpg)."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def run_fn_extraction():
    logger.info("=" * 60)
    logger.info("  🔬 Inference & IoU Diagnostic Engine  ")
    logger.info("=" * 60)

    # ── Infrastructure checks ──────────────────────────────────────────────────
    if not WEIGHTS_PATH.exists():
        logger.error(f"❌ Model weights not found: {WEIGHTS_PATH}")
        sys.exit(1)

    if not LABELS_DIR.exists():
        logger.error(f"❌ Labels directory not found: {LABELS_DIR}")
        sys.exit(1)

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    if not label_files:
        logger.error("❌ No label files found in val labels directory.")
        sys.exit(1)

    logger.info(f"✅ Loaded weights: {WEIGHTS_PATH.name}")
    logger.info(f"✅ Found {len(label_files)} validation label files")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("🧠 Loading YOLOv8 model...")
    model = YOLO(str(WEIGHTS_PATH))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"   Running on: {device.upper()}")

    # ── Main inference + IoU loop ──────────────────────────────────────────────
    fn_map = {}          # { str(image_path): [[x1,y1,x2,y2], ...] }
    total_gt       = 0
    total_fn       = 0
    skipped_images = 0

    logger.info("🚀 Starting False Negative extraction pass...")

    for label_path in tqdm(label_files, desc="Scanning val set", unit="img"):
        stem = label_path.stem

        # ── 1. Parse ground-truth boxes ────────────────────────────────────────
        gt_boxes = parse_yolo_label(label_path)

        # Edge case: image has zero annotated boxes — nothing to miss
        if gt_boxes.shape[0] == 0:
            continue

        total_gt += gt_boxes.shape[0]

        # ── 2. Find corresponding image ────────────────────────────────────────
        img_path = find_image(stem)
        if img_path is None:
            logger.warning(f"   ⚠️  Image not found for label: {stem} — skipping")
            skipped_images += 1
            continue

        # ── 3. Run model inference (no gradient tracking needed) ───────────────
        with torch.no_grad():
            results = model.predict(
                source=str(img_path),
                conf=CONF_THRESHOLD,
                verbose=False,   # Suppress per-image console spam
            )

        # Extract predicted boxes (xyxy pixel format)
        pred_boxes_np = results[0].boxes.xyxy.cpu()   # Tensor (M, 4)

        # ── 4. Vectorized IoU via torchvision.ops.box_iou ─────────────────────
        #    Returns matrix of shape (N_gt, M_pred).
        #    box_iou handles the empty-prediction edge case:
        #    if M=0, returns a (N, 0) tensor → max over dim=1 → all 0.0 → all FN
        iou_matrix = box_iou(gt_boxes, pred_boxes_np)  # (N_gt, M_pred)

        if iou_matrix.numel() == 0 or iou_matrix.shape[1] == 0:
            # Model made zero predictions — every GT box is a false negative
            max_iou_per_gt = torch.zeros(gt_boxes.shape[0])
        else:
            max_iou_per_gt = iou_matrix.max(dim=1).values  # (N_gt,)

        # ── 5. Apply FN threshold ──────────────────────────────────────────────
        fn_mask = max_iou_per_gt < FN_IOU_THRESHOLD   # Bool tensor
        fn_count = fn_mask.sum().item()

        if fn_count > 0:
            total_fn += fn_count
            fn_boxes = gt_boxes[fn_mask].tolist()  # List of [x1, y1, x2, y2]
            fn_map[str(img_path)] = fn_boxes

    # ── Write output JSON ──────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(fn_map, f, indent=2)

    # ── Summary report ─────────────────────────────────────────────────────────
    fn_rate = (total_fn / total_gt * 100) if total_gt > 0 else 0.0
    logger.info("=" * 60)
    logger.info("  📊 DIAGNOSTIC REPORT")
    logger.info("=" * 60)
    logger.info(f"   Total GT boxes scanned  : {total_gt:,}")
    logger.info(f"   Total False Negatives   : {total_fn:,}  ({fn_rate:.1f}% miss rate)")
    logger.info(f"   Images with ≥1 FN       : {len(fn_map):,}")
    logger.info(f"   Images skipped (no img) : {skipped_images}")
    logger.info(f"   IoU threshold used      : < {FN_IOU_THRESHOLD}")
    logger.info("-" * 60)
    logger.info(f"✅ Success! fn_coordinates.json written to:")
    logger.info(f"   {OUTPUT_PATH}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_fn_extraction()
