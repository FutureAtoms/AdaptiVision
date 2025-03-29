#!/usr/bin/env python
"""
Generate visualizations for AdaptiVision's Adaptive Context-Aware thresholding
"""
import argparse
import os
import time
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json
import io

from adaptivision import AdaptiVision

def parse_args():
    parser = argparse.ArgumentParser(description="Create visualizations for adaptive confidence thresholds")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--output-dir", type=str, default="results/visualizations", help="Directory for output images")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    return parser.parse_args()

def calculate_scene_complexity(detections, img_shape):
    """Calculate scene complexity based on detections"""
    if not detections or len(detections) == 0:
        return 0.3  # Default complexity
    
    # 1. Number of objects (normalized with log scale)
    num_objects = len(detections)
    norm_num_objects = min(1.0, np.log(num_objects + 1) / np.log(10))  # Log scale normalization
    
    # 2. Object size variance
    areas = []
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        areas.append((x2 - x1) * (y2 - y1))
    
    if areas:
        area_std = np.std(areas) / (img_shape[0] * img_shape[1])
        norm_size_variance = min(1.0, area_std * 10)  # Normalize to [0,1]
    else:
        norm_size_variance = 0.0
    
    # 3. Object density (total box area / image area)
    total_area = sum(areas)
    img_area = img_shape[0] * img_shape[1]
    density = min(1.0, total_area / img_area * 3)  # Normalize with ceiling at 1.0
    
    # Combine factors with weights
    complexity = 0.5 * norm_num_objects + 0.25 * norm_size_variance + 0.25 * density
    return complexity

def generate_heatmap(detections, img_shape, adaptive_threshold):
    """Generate a heatmap showing complexity regions"""
    # Create empty heatmap
    heatmap = np.zeros((img_shape[0], img_shape[1]), dtype=np.float32)
    
    # Add Gaussian for each detection
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = float(det[4])
        
        # Calculate center and size
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        # Sigma based on object size
        sigma_x = width / 3
        sigma_y = height / 3
        
        # Generate 2D Gaussian
        y_indices, x_indices = np.meshgrid(
            np.arange(img_shape[0]), 
            np.arange(img_shape[1]), 
            indexing='ij'
        )
        
        gaussian = np.exp(-((x_indices - center_x) ** 2 / (2 * sigma_x ** 2) + 
                           (y_indices - center_y) ** 2 / (2 * sigma_y ** 2)))
        
        # Add weighted gaussian to heatmap
        heatmap += gaussian * conf
    
    # Normalize
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    return heatmap

def adaptive_threshold_to_color(threshold, base_threshold=0.25):
    """Convert threshold to color (red for lower, green for higher)"""
    if threshold < base_threshold:
        # Lower threshold (more lenient) - reddish
        ratio = (base_threshold - threshold) / 0.15  # Assumes max reduction is 0.15
        ratio = min(1.0, max(0.0, ratio))
        return (255, int(255 * (1 - ratio)), int(255 * (1 - ratio)))
    else:
        # Higher threshold (more strict) - greenish
        ratio = (threshold - base_threshold) / 0.15  # Assumes max increase is 0.15
        ratio = min(1.0, max(0.0, ratio))
        return (int(255 * (1 - ratio)), 255, int(255 * (1 - ratio)))

def create_complexity_visualization(image, detections, complexity, adaptive_threshold, base_threshold, model_names):
    """Create visualization showing complexity analysis"""
    # Convert to PIL for drawing
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font, default to basic if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Generate heatmap
    heatmap = generate_heatmap(detections, image.shape[:2], adaptive_threshold)
    
    # Create heatmap overlay
    heatmap_rgb = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Blend with original image
    overlay = cv2.addWeighted(image, 0.7, heatmap_rgb, 0.3, 0)
    
    # Create info panel
    h, w = image.shape[:2]
    info_panel = np.ones((h, 300, 3), dtype=np.uint8) * 255
    
    # Add complexity gauge (0 to 1 scale)
    cv2.rectangle(info_panel, (50, 50), (250, 70), (200, 200, 200), -1)
    gauge_width = int(200 * complexity)
    
    # Determine gauge color based on complexity
    if complexity < 0.3:
        gauge_color = (0, 255, 0)  # Green for low complexity
    elif complexity < 0.6:
        gauge_color = (0, 255, 255)  # Yellow for medium 
    else:
        gauge_color = (0, 0, 255)  # Red for high complexity
    
    cv2.rectangle(info_panel, (50, 50), (50 + gauge_width, 70), gauge_color, -1)
    cv2.putText(info_panel, "Scene Complexity: {:.2f}".format(complexity), 
                (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add adaptive threshold info
    threshold_color = adaptive_threshold_to_color(adaptive_threshold, base_threshold)
    cv2.putText(info_panel, "Base Threshold: {:.2f}".format(base_threshold), 
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(info_panel, "Adaptive Threshold: {:.3f}".format(adaptive_threshold), 
                (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threshold_color[::-1], 2)
    
    adjustment = adaptive_threshold - base_threshold
    if adjustment < 0:
        adjustment_text = f"Lowered by: {abs(adjustment):.3f}"
    else:
        adjustment_text = f"Raised by: {adjustment:.3f}"
    
    cv2.putText(info_panel, adjustment_text, 
                (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, threshold_color[::-1], 1)
    
    # Add legend for detections
    cv2.putText(info_panel, "Detections:", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    y_offset = 200
    for i, det in enumerate(detections):
        if len(det) >= 6:  # class id is included
            class_id = int(det[5])
            class_name = model_names[class_id] if model_names else f"Class {class_id}"
            confidence = det[4]
            
            # Determine if this detection would pass standard vs adaptive threshold
            passes_standard = confidence >= base_threshold
            passes_adaptive = confidence >= adaptive_threshold
            
            status_color = (0, 0, 0)  # Default black
            status_text = ""
            
            if passes_standard and passes_adaptive:
                status_color = (0, 128, 0)  # Dark green
                status_text = "Detected by both"
            elif not passes_standard and passes_adaptive:
                status_color = (255, 0, 0)  # Red
                status_text = "Rescued by adaptive"
            elif passes_standard and not passes_adaptive:
                status_color = (0, 0, 255)  # Blue
                status_text = "Filtered by adaptive"
            
            cv2.putText(info_panel, f"{class_name}: {confidence:.3f}", 
                        (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if status_text:
                cv2.putText(info_panel, status_text, 
                            (200, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
            
            y_offset += 20
    
    # Combine image and panel
    combined = np.hstack((overlay, info_panel))
    
    return combined

def create_threshold_map_visualization(image, detections, base_threshold=0.25):
    """Create a visualization of the adaptive threshold map across the image"""
    h, w = image.shape[:2]
    
    # Create a grid of points across the image
    grid_size = 20
    x_points = np.linspace(0, w, grid_size)
    y_points = np.linspace(0, h, grid_size)
    
    # Create empty threshold map
    threshold_map = np.zeros((h, w), dtype=np.float32)
    
    # Calculate local complexity and threshold for each grid point
    for y in range(grid_size):
        for x in range(grid_size):
            # Calculate region
            x1 = int(max(0, x_points[x] - w/grid_size))
            y1 = int(max(0, y_points[y] - h/grid_size))
            x2 = int(min(w, x_points[x] + w/grid_size))
            y2 = int(min(h, y_points[y] + h/grid_size))
            
            # Find detections in this region
            local_dets = []
            for det in detections:
                det_x1, det_y1, det_x2, det_y2 = map(int, det[:4])
                det_center_x = (det_x1 + det_x2) // 2
                det_center_y = (det_y1 + det_y2) // 2
                
                if (x1 <= det_center_x <= x2) and (y1 <= det_center_y <= y2):
                    local_dets.append(det)
            
            # Calculate local complexity
            local_complexity = calculate_scene_complexity(local_dets, (y2-y1, x2-x1))
            
            # Adjust threshold based on complexity
            threshold_adjustment = max_adjust - local_complexity * (max_adjust - min_adjust)
            local_threshold = max(0.1, min(0.4, base_threshold + threshold_adjustment))
            
            # Fill region with this threshold
            threshold_map[y1:y2, x1:x2] = local_threshold
    
    # Smooth the threshold map
    threshold_map = cv2.GaussianBlur(threshold_map, (15, 15), 0)
    
    # Create a custom colormap
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red -> Yellow -> Green
    cmap = LinearSegmentedColormap.from_list("adaptive_threshold_map", colors)
    
    # Normalize threshold map for visualization
    norm_map = (threshold_map - 0.1) / 0.3  # Map from [0.1, 0.4] to [0, 1]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot threshold map
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), alpha=0.7)
    im = plt.imshow(norm_map, cmap=cmap, alpha=0.5)
    plt.colorbar(im, label="Confidence Threshold")
    plt.title("Adaptive Threshold Map")
    plt.axis('off')
    
    # Create buffer for image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    threshold_map_img = np.array(Image.open(buf))
    plt.close()
    
    return threshold_map_img

# Min and max adjustment constants
min_adjust = -0.15  # Maximum reduction for complex scenes
max_adjust = 0.05   # Maximum increase for simple scenes

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create detector
    detector = AdaptiVision(
        model_path=args.weights,
        device=args.device,
        conf_threshold=0.25,  # Standard threshold
        enable_adaptive_confidence=True,
        context_aware=True
    )
    
    # Load image
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: Unable to load image {args.image}")
        return
    
    # Run initial detection with low threshold to get all potential objects
    results = detector.predict(args.image, conf_threshold=0.1)
    
    if not results or len(results[0]['boxes']) == 0:
        print("No detections found. Try a different image or lower the confidence threshold.")
        return
    
    detections = []
    model_names = {}
    
    # Extract detections from results
    for result in results:
        boxes = result['boxes']
        scores = result['scores']
        labels = result['labels']
        class_names = result['class_names']
        
        for i, (box, score, label, name) in enumerate(zip(boxes, scores, labels, class_names)):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, score, label])
            model_names[label] = name
    
    # Standard base threshold
    base_threshold = 0.25
    
    # Calculate scene complexity
    complexity = calculate_scene_complexity(detections, img.shape[:2])
    
    # Calculate adaptive threshold
    threshold_adjustment = max_adjust - complexity * (max_adjust - min_adjust)
    adaptive_threshold = max(0.1, min(0.4, base_threshold + threshold_adjustment))
    
    print(f"Scene complexity: {complexity:.3f}")
    print(f"Base threshold: {base_threshold:.3f}")
    print(f"Adaptive threshold: {adaptive_threshold:.3f}")
    
    # Create complexity visualization
    complexity_viz = create_complexity_visualization(
        img.copy(), detections, complexity, adaptive_threshold, base_threshold, model_names
    )
    
    # Save complexity visualization
    output_path = os.path.join(args.output_dir, f"complexity_{os.path.basename(args.image)}")
    cv2.imwrite(output_path, complexity_viz)
    print(f"Saved complexity visualization to {output_path}")
    
    # Try to create threshold map visualization
    try:
        threshold_map_viz = create_threshold_map_visualization(
            img.copy(), detections, base_threshold
        )
        
        # Save threshold map visualization
        map_output_path = os.path.join(args.output_dir, f"threshold_map_{os.path.basename(args.image)}")
        cv2.imwrite(map_output_path, threshold_map_viz)
        print(f"Saved threshold map visualization to {map_output_path}")
    except Exception as e:
        print(f"Skipping threshold map visualization: {e}")
    
    # Prepare detection data for metadata
    detection_data = []
    for det in detections:
        # Convert boolean to int to avoid JSON serialization issues
        passes_standard = 1 if det[4] >= base_threshold else 0
        passes_adaptive = 1 if det[4] >= adaptive_threshold else 0
        
        class_id = int(det[5])
        class_name = model_names.get(class_id, f"Class {class_id}")
        
        detection_data.append({
            "bbox": [float(x) for x in det[:4]],
            "confidence": float(det[4]),
            "class_id": class_id,
            "class_name": class_name,
            "passes_standard": passes_standard,
            "passes_adaptive": passes_adaptive
        })
    
    # Save detection results with metadata
    output_metadata = {
        "image_path": args.image,
        "complexity": float(complexity),
        "base_threshold": float(base_threshold),
        "adaptive_threshold": float(adaptive_threshold),
        "detections": detection_data
    }
    
    json_path = os.path.join(args.output_dir, f"metadata_{os.path.basename(args.image).split('.')[0]}.json")
    with open(json_path, 'w') as f:
        json.dump(output_metadata, f, indent=2)
    
    print(f"Saved metadata to {json_path}")
    print("Done!")

if __name__ == "__main__":
    main() 