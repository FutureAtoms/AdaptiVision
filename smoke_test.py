"""
Simple smoke test for AdaptiVision
Works on Windows, Mac, and Linux
"""
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from adaptivision import AdaptiVision

def main():
    # Use pathlib for cross-platform paths
    weights_path = Path("weights") / "model_n.pt"
    image_path = Path("samples") / "bus.jpg"
    output_path = Path("results") / "smoke_test.jpg"

    # Create results directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AdaptiVision Smoke Test")
    print("=" * 60)
    print(f"Weights: {weights_path}")
    print(f"Image: {image_path}")
    print(f"Output: {output_path}")
    print()

    # Check files exist
    if not weights_path.exists():
        print(f"ERROR: Model weights not found at {weights_path}")
        return 1

    if not image_path.exists():
        print(f"ERROR: Test image not found at {image_path}")
        return 1

    print("Initializing AdaptiVision...")
    detector = AdaptiVision(
        model_path=str(weights_path),
        device='auto',
        conf_threshold=0.25,
        iou_threshold=0.45,
        enable_adaptive_confidence=True,
        context_aware=True,
        enable_postprocess_filter=True
    )

    print("Running detection...")
    results = detector.predict(str(image_path))
    detection_data = results[0]

    print(f"Detected {len(detection_data.get('boxes', []))} objects")

    print("Saving visualization...")
    detector.visualize(str(image_path), detection_data, str(output_path))

    print()
    print("=" * 60)
    print("SUCCESS! Smoke test passed.")
    print(f"Check output at: {output_path}")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
