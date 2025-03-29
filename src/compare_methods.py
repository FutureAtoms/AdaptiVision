#!/usr/bin/env python
"""
Compare standard object detection with Adaptive Context-Aware thresholding
"""
import os
import argparse
import time
import cv2
import numpy as np
from pathlib import Path

from adaptivision import AdaptiVision

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Standard vs Adaptive Context-Aware Thresholding")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Base confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--output-dir", type=str, default="results/comparison", help="Directory for output images")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, mps)")
    return parser.parse_args()

def run_standard(image_path, weights_path, device, conf_threshold, iou_threshold):
    """Run inference with standard confidence threshold"""
    print("\n=== Running Standard Object Detection ===")
    
    # Create detector with standard settings
    detector = AdaptiVision(
        model_path=weights_path,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        enable_adaptive_confidence=False,
        context_aware=False
    )
    
    # Run inference
    start_time = time.time()
    results = detector.predict(image_path, verbose=True)
    inference_time = time.time() - start_time
    
    print(f"Standard detection time: {inference_time:.4f} seconds")
    
    # Return detection results and image
    if results and len(results) > 0:
        detections = results[0]
        
        # Load image for visualization
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
            
        return image, detections, inference_time
    else:
        print("Standard detection: No results found")
        return None, None, inference_time

def run_adaptive(image_path, weights_path, device, conf_threshold, iou_threshold):
    """Run inference with adaptive confidence threshold"""
    print("\n=== Running Adaptive Context-Aware Detection ===")
    
    # Create detector with adaptive settings
    detector = AdaptiVision(
        model_path=weights_path,
        device=device,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        enable_adaptive_confidence=True,
        context_aware=True
    )
    
    # Run inference
    start_time = time.time()
    results = detector.predict(image_path, verbose=True)
    inference_time = time.time() - start_time
    
    print(f"Adaptive detection time: {inference_time:.4f} seconds")
    
    # Return detection results and image
    if results and len(results) > 0:
        detections = results[0]
        
        # Load image for visualization
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
            
        # Print adaptive details
        if 'scene_complexity' in detections and 'adaptive_threshold' in detections:
            print(f"\nAdaptive confidence details:")
            print(f"  Scene complexity: {detections['scene_complexity']:.2f}")
            print(f"  Adaptive threshold: {detections['adaptive_threshold']:.3f}")
            
        return image, detections, inference_time
    else:
        print("Adaptive detection: No results found")
        return None, None, inference_time

def create_comparison_image(original_img, standard_detections, adaptive_detections, base_threshold, output_path):
    """Create side-by-side comparison image with detection results"""
    if original_img is None:
        print("Cannot create comparison: No image provided")
        return None
        
    # Create copies for drawing
    standard_img = original_img.copy()
    adaptive_img = original_img.copy()
    
    # Get image dimensions
    h, w, _ = original_img.shape
    
    # Generate colors for visualization
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(80)]
    
    # Draw standard detections
    if standard_detections is not None and len(standard_detections['boxes']) > 0:
        boxes = standard_detections['boxes']
        scores = standard_detections['scores']
        labels = standard_detections['labels']
        class_names = standard_detections['class_names']
        
        for i, (box, score, label, name) in enumerate(zip(boxes, scores, labels, class_names)):
            color = colors[label % len(colors)]
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(standard_img, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label_text = f"{name} {score:.2f}"
            
            # Draw label
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(standard_img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(standard_img, label_text, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw adaptive detections
    if adaptive_detections is not None and len(adaptive_detections['boxes']) > 0:
        boxes = adaptive_detections['boxes']
        scores = adaptive_detections['scores']
        labels = adaptive_detections['labels']
        class_names = adaptive_detections['class_names']
        
        for i, (box, score, label, name) in enumerate(zip(boxes, scores, labels, class_names)):
            color = colors[label % len(colors)]
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(adaptive_img, (x1, y1), (x2, y2), color, 2)
            
            # Create label text
            label_text = f"{name} {score:.2f}"
            
            # Draw label
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(adaptive_img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
            cv2.putText(adaptive_img, label_text, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Create headers
    standard_header = np.zeros((50, w, 3), dtype=np.uint8)
    adaptive_header = np.zeros((50, w, 3), dtype=np.uint8)
    
    # Add header text
    standard_title = f"Standard Detection (threshold: {base_threshold:.2f})"
    cv2.putText(standard_header, standard_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add adaptive threshold info
    adaptive_title = "Adaptive Context-Aware Detection"
    cv2.putText(adaptive_header, adaptive_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add threshold and complexity info
    if adaptive_detections and 'adaptive_threshold' in adaptive_detections and 'scene_complexity' in adaptive_detections:
        adaptive_info = f"Complexity: {adaptive_detections['scene_complexity']:.2f}, Threshold: {adaptive_detections['adaptive_threshold']:.2f}"
        cv2.putText(
            adaptive_img, 
            adaptive_info, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 200, 255), 
            2
        )
    
    # Combine images
    standard_with_header = np.vstack([standard_header, standard_img])
    adaptive_with_header = np.vstack([adaptive_header, adaptive_img])
    
    # Create side-by-side comparison
    comparison = np.hstack([standard_with_header, adaptive_with_header])
    
    # Add detection counts
    standard_count = len(standard_detections['boxes']) if standard_detections else 0
    adaptive_count = len(adaptive_detections['boxes']) if adaptive_detections else 0
    
    # Add comparison stats
    summary_height = 50
    summary = np.zeros((summary_height, comparison.shape[1], 3), dtype=np.uint8)
    
    # Add stats text
    stats_text = f"Objects Detected: Standard: {standard_count}, Adaptive: {adaptive_count}"
    cv2.putText(summary, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add summary to comparison
    comparison = np.vstack([comparison, summary])
    
    # Save comparison image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Comparison image saved to {output_path}")
    
    return comparison

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run standard detection
    standard_img, standard_detections, standard_time = run_standard(
        args.image, args.weights, args.device, args.conf, args.iou
    )
    
    # Run adaptive detection
    adaptive_img, adaptive_detections, adaptive_time = run_adaptive(
        args.image, args.weights, args.device, args.conf, args.iou
    )
    
    # Create comparison image
    if standard_img is not None and adaptive_img is not None:
        # Get base filename
        base_filename = os.path.basename(args.image)
        output_path = os.path.join(args.output_dir, f"comparison_{base_filename}")
        
        # Create and save comparison
        create_comparison_image(
            standard_img, 
            standard_detections, 
            adaptive_detections, 
            args.conf,
            output_path
        )
        
        # Print performance comparison
        standard_count = len(standard_detections['boxes']) if standard_detections else 0
        adaptive_count = len(adaptive_detections['boxes']) if adaptive_detections else 0
        
        print("\n=== Performance Comparison ===")
        print(f"Standard Detection: {standard_count} objects in {standard_time:.4f}s")
        print(f"Adaptive Detection: {adaptive_count} objects in {adaptive_time:.4f}s")
        
        if 'adaptive_threshold' in adaptive_detections:
            adaptive_thresh = adaptive_detections['adaptive_threshold']
            threshold_diff = adaptive_thresh - args.conf
            print(f"Threshold adaptation: {args.conf:.2f} -> {adaptive_thresh:.2f} ({threshold_diff:+.2f})")
    else:
        print("Could not create comparison due to missing results")

if __name__ == "__main__":
    main() 