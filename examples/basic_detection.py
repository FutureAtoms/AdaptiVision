#!/usr/bin/env python
"""
Basic example of using AdaptiVision for object detection with adaptive thresholding.
"""
import os
import argparse
import time
import cv2
from pathlib import Path
import sys

# Add the parent directory to the path to import the AdaptiVision module
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.adaptivision import AdaptiVision

def parse_args():
    parser = argparse.ArgumentParser(description="Run AdaptiVision object detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="results/detection.jpg", help="Path to output image")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Base confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    parser.add_argument("--disable-adaptive", action="store_true", help="Disable adaptive confidence")
    parser.add_argument("--disable-context", action="store_true", help="Disable context-aware reasoning")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Print configuration
    print(f"Running AdaptiVision with the following configuration:")
    print(f"  - Input image: {args.image}")
    print(f"  - Model weights: {args.weights}")
    print(f"  - Base confidence threshold: {args.conf_thres}")
    print(f"  - IoU threshold: {args.iou_thres}")
    print(f"  - Device: {args.device}")
    print(f"  - Adaptive confidence: {'Disabled' if args.disable_adaptive else 'Enabled'}")
    print(f"  - Context-aware reasoning: {'Disabled' if args.disable_context else 'Enabled'}")
    
    # Initialize AdaptiVision detector
    start_init = time.time()
    detector = AdaptiVision(
        model_path=args.weights,
        device=args.device,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        enable_adaptive_confidence=not args.disable_adaptive,
        context_aware=not args.disable_context
    )
    init_time = time.time() - start_init
    print(f"Detector initialized in {init_time:.2f} seconds")
    
    # Run detection
    start_detect = time.time()
    results = detector.predict(args.image)
    detection_time = time.time() - start_detect
    print(f"Detection completed in {detection_time:.3f} seconds")
    
    # Print results summary
    if results and len(results) > 0:
        result = results[0]  # First result
        num_objects = len(result['boxes'])
        print(f"Found {num_objects} objects")
        
        # Print detected classes and counts
        classes = {}
        for i, label in enumerate(result['labels']):
            class_name = result['class_names'][i]
            conf = result['scores'][i]
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(conf)
        
        print("\nObject classes:")
        for class_name, confs in classes.items():
            avg_conf = sum(confs) / len(confs)
            print(f"  - {class_name}: {len(confs)} instances (avg conf: {avg_conf:.3f})")
            
        # Print adaptive threshold if enabled
        if not args.disable_adaptive and 'adaptive_threshold' in result:
            adaptive_threshold = result['adaptive_threshold']
            complexity = result['complexity']
            adjustment = adaptive_threshold - args.conf_thres
            print(f"\nScene complexity: {complexity:.3f}")
            print(f"Adaptive threshold: {adaptive_threshold:.3f} " +
                  f"({'decreased' if adjustment < 0 else 'increased'} by {abs(adjustment):.3f})")
    else:
        print("No objects detected")
    
    # Save detection visualization
    output_path = detector.visualize_detections(args.image, args.output)
    print(f"\nResults saved to {output_path}")
    
    # Display image if not in headless environment
    try:
        img = cv2.imread(output_path)
        window_name = "AdaptiVision Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        print("\nPress any key to exit")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        # Skip visualization if running headless
        pass

if __name__ == "__main__":
    main() 