import sys
import pandas as pd
from pathlib import Path

# Fix for Windows console emoji printing (UnicodeEncodeError)
if sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

def run_baseline_autopsy(csv_path: str):
    path = Path(csv_path)
    if not path.exists():
        print(f"❌ Error: Could not find telemetry at {path.resolve()}")
        return

    # Load the raw telemetry
    df = pd.read_csv(path)
    
    # YOLOv8 leaves terrible whitespace in its CSV headers. We clean it.
    df.columns = df.columns.str.strip()

    # Extract the critical metrics
    best_epoch = df['metrics/mAP50-95(B)'].idxmax()
    best_map50_95 = df.loc[best_epoch, 'metrics/mAP50-95(B)']
    best_map50 = df.loc[best_epoch, 'metrics/mAP50(B)']
    
    # Calculate the Resolution Gap
    resolution_gap = best_map50 - best_map50_95

    print("\n" + "="*50)
    print(" 🩸 BASELINE ERROR AUTOPSY REPORT")
    print("="*50)
    print(f"✅ Optimal Epoch Hit: {best_epoch + 1}")
    print(f"🎯 Peak mAP@50 (Loose Box):  {best_map50:.4f}")
    print(f"🎯 Peak mAP@50-95 (Strict Box): {best_map50_95:.4f}")
    print("-" * 50)
    print(f"⚠️ Resolution Penalty Gap: {resolution_gap:.4f}")
    
    if resolution_gap > 0.30:
        print("\n🚨 CRITICAL DIAGNOSIS: Massive Resolution Bottleneck Detected.")
        print("The model easily finds the object but mathematically cannot draw tight")
        print("boundaries due to 640px compression destroying edge pixels.")
        print("RECOMMENDATION: Scale architecture to 1024px.")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Dynamically point to safely routed CSV relative to this script
    script_dir = Path(__file__).resolve().parent
    default_csv = script_dir.parent / "outputs" / "yolov8_baseline_640" / "results.csv"
    run_baseline_autopsy(str(default_csv))