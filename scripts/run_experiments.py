#!/usr/bin/env python3
import os
import sys
import json
import time
import csv
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image, UnidentifiedImageError

# Add parent directory to path to import AdaptiVision modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AdaptiVision modules
from src.adaptivision import AdaptiVision

# --- ADDITION: Add src directory to path for internal imports --- 
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)
# --- END ADDITION ---

def parse_args():
    parser = argparse.ArgumentParser(description="Run AdaptiVision experiments")
    parser.add_argument("--data", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save results")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    return parser.parse_args()

def run_experiments(data_dir, output_dir, model_path="weights/model_n.pt", device="auto", reanalyze_only=False):
    """
    Run comprehensive experiments comparing standard YOLO with AdaptiVision
    
    Args:
        data_dir: Directory containing input images
        output_dir: Directory to save results
        model_path: Path to YOLO model weights
        device: Device to run inference on
        reanalyze_only: Flag to reanalyze existing results
    """
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "standard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "adaptive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analytics"), exist_ok=True)
    
    # --- Check if re-analyzing existing results ---
    detailed_results_path = os.path.join(output_dir, "detailed_results.json")
    if reanalyze_only and os.path.exists(detailed_results_path):
        print(f"Re-analyzing existing results from: {detailed_results_path}")
        with open(detailed_results_path, "r") as f:
            results = json.load(f)
        # Skip directly to analytics
        generate_analytics(results, output_dir)
        print(f"Re-analysis complete. Analytics updated in {output_dir}")
        return
    elif reanalyze_only:
        print(f"Error: Cannot re-analyze. Results file not found: {detailed_results_path}")
        return
    # --- End Re-analysis Check ---
    
    # Load metadata if available
    metadata_path = os.path.join(data_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        image_files = [os.path.join(data_dir, item["filename"]) for item in metadata["files"]]
    else:
        # Get all image files
        image_files = list(Path(data_dir).glob("*.jpg"))
        image_files = [str(f) for f in image_files]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Using device: {device}")
    
    # Initialize detectors
    standard_detector = AdaptiVision(
        model_path=model_path,
        device=device,
        enable_adaptive_confidence=False,
        context_aware=False
    )
    
    adaptive_detector = AdaptiVision(
        model_path=model_path, 
        device=device,
        enable_adaptive_confidence=True,
        context_aware=True
    )
    
    # Prepare data collection
    results = []
    
    # Process each image
    total_images = len(image_files)
    for i, img_path in enumerate(image_files):
        # Print progress indicator
        print(f"\n--- Processing image {i+1}/{total_images}: {os.path.basename(img_path)} ---")
        
        # Load image data once
        try:
            # Try loading with OpenCV first
            original_image = cv2.imread(img_path)
            if original_image is None:
                # Try PIL fallback
                pil_img = Image.open(img_path)
                pil_img.load()
                np_img = np.array(pil_img)
                if len(np_img.shape) == 2: # Grayscale
                    original_image = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
                elif np_img.shape[2] == 4: # RGBA
                    original_image = cv2.cvtColor(np_img, cv2.COLOR_RGBA2BGR)
                elif np_img.shape[2] == 3: # RGB
                    original_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                else:
                    raise ValueError(f"Unsupported channels: {np_img.shape[2]}")
        except Exception as load_e:
            print(f"Error loading image {img_path}: {load_e}. Skipping this image.")
            # Add error entry to results and continue
            results.append({
                "filename": os.path.basename(img_path),
                "image_path": img_path,
                "standard_detection": {"success": False, "error": f"Image load error: {load_e}"},
                "adaptive_detection": {"success": False, "error": f"Image load error: {load_e}"},
                "comparison": {}, "visualization": {}
            })
            continue # Skip to the next image
        
        img_filename = os.path.basename(img_path)
        img_stem = os.path.splitext(img_filename)[0]
        
        # Output paths
        standard_img_path = os.path.join(output_dir, "standard", f"standard_{img_filename}")
        adaptive_img_path = os.path.join(output_dir, "adaptive", f"adaptive_{img_filename}")
        comparison_path = os.path.join(output_dir, "comparisons", f"comparison_{img_filename}")
        visualization_dir = os.path.join(output_dir, "visualizations", img_stem)
        os.makedirs(visualization_dir, exist_ok=True)
        
        # Record data for this image
        img_data = {
            "filename": img_filename,
            "image_path": img_path,
            "standard_detection": {},
            "adaptive_detection": {},
            "comparison": {},
            "visualization": {}
        }
        
        # Run standard detection
        start_time = time.time()
        try:
            # Pass the loaded image array to predict
            standard_results = standard_detector.predict(original_image)
            standard_time = time.time() - start_time
            
            # Get object counts
            if standard_results and len(standard_results) > 0:
                # Extract detections
                standard_boxes = standard_results[0].get('boxes', [])
                standard_scores = standard_results[0].get('scores', [])
                standard_labels = standard_results[0].get('labels', [])
                standard_class_names = standard_results[0].get('class_names', [])
                
                # Count objects by class
                standard_counts = {}
                for name in standard_class_names:
                    if name in standard_counts:
                        standard_counts[name] += 1
                    else:
                        standard_counts[name] = 1
                
                # Save detection image
                standard_detector.visualize(original_image, standard_results[0], standard_img_path)
                
                img_data["standard_detection"] = {
                    "success": True,
                    "detection_time": standard_time,
                    "object_count": len(standard_boxes),
                    "objects_by_class": standard_counts,
                    "confidence_scores": [float(s) for s in standard_scores] if len(standard_scores) > 0 else [],
                    "output_path": standard_img_path
                }
            else:
                img_data["standard_detection"] = {
                    "success": False,
                    "detection_time": standard_time,
                    "error": "No detections found"
                }
        except Exception as e:
            img_data["standard_detection"] = {
                "success": False,
                "detection_time": time.time() - start_time,
                "error": str(e)
            }
        
        # Run adaptive detection
        start_time = time.time()
        try:
            # Pass the loaded image array to predict
            adaptive_results = adaptive_detector.predict(original_image)
            adaptive_time = time.time() - start_time
            
            # Get object counts
            if adaptive_results and len(adaptive_results) > 0:
                # Extract detections
                adaptive_boxes = adaptive_results[0].get('boxes', [])
                adaptive_scores = adaptive_results[0].get('scores', [])
                adaptive_labels = adaptive_results[0].get('labels', [])
                adaptive_class_names = adaptive_results[0].get('class_names', [])
                
                # Get scene complexity and adaptive threshold
                scene_complexity = adaptive_results[0].get('scene_complexity', 0)
                adaptive_threshold = adaptive_results[0].get('adaptive_threshold', 0.25)
                
                # Count objects by class
                adaptive_counts = {}
                for name in adaptive_class_names:
                    if name in adaptive_counts:
                        adaptive_counts[name] += 1
                    else:
                        adaptive_counts[name] = 1
                
                # Save detection image
                adaptive_detector.visualize(original_image, adaptive_results[0], adaptive_img_path)
                
                img_data["adaptive_detection"] = {
                    "success": True,
                    "detection_time": adaptive_time,
                    "object_count": len(adaptive_boxes),
                    "objects_by_class": adaptive_counts,
                    "confidence_scores": [float(s) for s in adaptive_scores] if len(adaptive_scores) > 0 else [],
                    "scene_complexity": float(scene_complexity),
                    "adaptive_threshold": float(adaptive_threshold),
                    "output_path": adaptive_img_path
                }
                
                # Create visualization of adaptive thresholding
                try:
                    # Import the visualization functions
                    from src.create_visualizations import create_complexity_visualization, create_threshold_map_visualization
                    
                    # Get necessary data (make sure adaptive_results[0] exists)
                    adaptive_result_data = adaptive_results[0]
                    all_detections_for_vis = np.hstack((
                        adaptive_result_data.get('boxes', np.array([])),
                        adaptive_result_data.get('scores', np.array([])).reshape(-1, 1),
                        adaptive_result_data.get('labels', np.array([])).reshape(-1, 1)
                    )) if len(adaptive_result_data.get('boxes', [])) > 0 else np.array([])
                    model_class_names = adaptive_detector.model.names if hasattr(adaptive_detector, 'model') else []
                    base_threshold_val = adaptive_result_data.get('base_threshold', 0.25)

                    # Complexity visualization
                    complexity_path = os.path.join(visualization_dir, f"complexity_{img_filename}")
                    vis_complex_img = create_complexity_visualization(
                        image=original_image.copy(), # Pass image array
                        detections=all_detections_for_vis, 
                        complexity=scene_complexity, 
                        adaptive_threshold=adaptive_threshold,
                        base_threshold=base_threshold_val,
                        model_names=model_class_names
                    )
                    if vis_complex_img is not None:
                        cv2.imwrite(complexity_path, vis_complex_img)
                    
                    # Threshold map visualization
                    threshold_path = os.path.join(visualization_dir, f"threshold_map_{img_filename}")
                    vis_thresh_img = create_threshold_map_visualization(
                        image=original_image.copy(), # Pass image array
                        detections=all_detections_for_vis,
                        base_threshold=base_threshold_val,
                        adaptive_threshold=adaptive_threshold # Pass the final adaptive threshold
                    )
                    if vis_thresh_img is not None:
                        cv2.imwrite(threshold_path, vis_thresh_img)
                    
                    # Save metadata
                    metadata_path = os.path.join(visualization_dir, f"metadata_{img_stem}.json")
                    with open(metadata_path, "w") as f:
                        json.dump({
                            "scene_complexity": float(scene_complexity),
                            "base_threshold": 0.25,
                            "adaptive_threshold": float(adaptive_threshold)
                        }, f, indent=2)
                    
                    img_data["visualization"] = {
                        "success": True,
                        "complexity_path": complexity_path,
                        "threshold_path": threshold_path,
                        "metadata_path": metadata_path
                    }
                except Exception as e:
                    img_data["visualization"] = {
                        "success": False,
                        "error": str(e)
                    }
            else:
                img_data["adaptive_detection"] = {
                    "success": False,
                    "detection_time": adaptive_time,
                    "error": "No detections found"
                }
        except Exception as e:
            img_data["adaptive_detection"] = {
                "success": False,
                "detection_time": time.time() - start_time,
                "error": str(e)
            }
        
        # Create comparison image if both detections succeeded
        if (img_data["standard_detection"].get("success", False) and 
            img_data["adaptive_detection"].get("success", False)):
            try:
                from src.compare_methods import create_comparison_image
                
                # Get result data
                standard_result = standard_results[0]
                adaptive_result = adaptive_results[0]
                
                # Get base threshold used
                base_threshold_used = adaptive_result.get('base_threshold', 0.25)
                
                # Convert BGR image (from cv2.imread or fallback) to RGB for the comparison function
                original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                
                # Create comparison image
                comparison_vis_img = create_comparison_image(
                    original_image_rgb,   # Pass RGB image array
                    standard_result,      
                    adaptive_result,      
                    base_threshold_used,  
                    comparison_path       
                )
                
                # Add comparison info
                base_threshold = base_threshold_used # Use the actual base threshold for consistency
                adaptive_threshold = float(adaptive_result.get("adaptive_threshold", base_threshold))
                threshold_diff = adaptive_threshold - base_threshold
                
                standard_count = img_data["standard_detection"]["object_count"]
                adaptive_count = img_data["adaptive_detection"]["object_count"]
                count_diff = adaptive_count - standard_count
                
                standard_time = img_data["standard_detection"]["detection_time"]
                adaptive_time = img_data["adaptive_detection"]["detection_time"]
                
                img_data["comparison"] = {
                    "success": True,
                    "output_path": comparison_path,
                    "standard_count": standard_count,
                    "adaptive_count": adaptive_count,
                    "count_difference": count_diff,
                    "standard_time": standard_time,
                    "adaptive_time": adaptive_time,
                    "speed_improvement": standard_time / adaptive_time if adaptive_time > 0 else 0,
                    "base_threshold": base_threshold,
                    "adaptive_threshold": adaptive_threshold,
                    "threshold_difference": threshold_diff
                }
            except Exception as e:
                img_data["comparison"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Add to results
        results.append(img_data)
    
    # Save detailed results to JSON
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Create CSV for easier analysis
    with open(os.path.join(output_dir, "summary_results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Filename", "Scene Complexity", "Base Threshold", "Adaptive Threshold", 
            "Threshold Change", "Standard Objects", "Adaptive Objects", "Object Count Diff",
            "Standard Time (s)", "Adaptive Time (s)", "Speed Improvement"
        ])
        
        for img_data in results:
            if not (img_data["standard_detection"].get("success", False) and 
                   img_data["adaptive_detection"].get("success", False)):
                continue
                
            filename = img_data["filename"]
            scene_complexity = img_data["adaptive_detection"].get("scene_complexity", 0)
            base_threshold = 0.25
            adaptive_threshold = img_data["adaptive_detection"].get("adaptive_threshold", base_threshold)
            threshold_change = adaptive_threshold - base_threshold
            
            standard_objects = img_data["standard_detection"].get("object_count", 0)
            adaptive_objects = img_data["adaptive_detection"].get("object_count", 0)
            object_diff = adaptive_objects - standard_objects
            
            standard_time = img_data["standard_detection"].get("detection_time", 0)
            adaptive_time = img_data["adaptive_detection"].get("detection_time", 0)
            speed_improvement = standard_time / adaptive_time if adaptive_time > 0 else 0
            
            writer.writerow([
                filename, scene_complexity, base_threshold, adaptive_threshold,
                threshold_change, standard_objects, adaptive_objects, object_diff,
                standard_time, adaptive_time, speed_improvement
            ])
    
    # Generate analytics and visualizations
    generate_analytics(results, output_dir)
    
    print(f"Experiment completed. Results saved to {output_dir}")

def generate_analytics(results, output_dir):
    """Generate analytics and visualizations from experiment results"""
    analytics_dir = os.path.join(output_dir, "analytics")
    
    # Extract key data for analysis
    scene_complexities = []
    threshold_changes = []
    standard_objects = []
    adaptive_objects = []
    standard_times = []
    adaptive_times = []
    speed_improvements = []
    filenames = []
    
    # Count successes and failures
    success_count = 0
    failure_count = 0
    
    for img_data in results:
        if not (img_data["standard_detection"].get("success", False) and 
               img_data["adaptive_detection"].get("success", False)):
            failure_count += 1
            continue
        
        success_count += 1
        filenames.append(img_data["filename"])
        scene_complexities.append(img_data["adaptive_detection"].get("scene_complexity", 0))
        
        base_threshold = 0.25
        adaptive_threshold = img_data["adaptive_detection"].get("adaptive_threshold", base_threshold)
        threshold_changes.append(adaptive_threshold - base_threshold)
        
        standard_objects.append(img_data["standard_detection"].get("object_count", 0))
        adaptive_objects.append(img_data["adaptive_detection"].get("object_count", 0))
        
        standard_times.append(img_data["standard_detection"].get("detection_time", 0))
        adaptive_times.append(img_data["adaptive_detection"].get("detection_time", 0))
        
        if adaptive_times[-1] > 0:
            speed_improvements.append(standard_times[-1] / adaptive_times[-1])
        else:
            speed_improvements.append(0)
    
    # Basic summary statistics
    summary = {
        "total_images": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "avg_scene_complexity": np.mean(scene_complexities) if scene_complexities else 0,
        "avg_threshold_change": np.mean(threshold_changes) if threshold_changes else 0,
        "avg_standard_objects": np.mean(standard_objects) if standard_objects else 0,
        "avg_adaptive_objects": np.mean(adaptive_objects) if adaptive_objects else 0,
        "avg_standard_time": np.mean(standard_times) if standard_times else 0,
        "avg_adaptive_time": np.mean(adaptive_times) if adaptive_times else 0,
        "avg_speed_improvement": np.mean(speed_improvements) if speed_improvements else 0,
    }
    
    # Save summary statistics
    with open(os.path.join(analytics_dir, "summary_statistics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots if we have data
    if success_count > 0:
        # 1. Scene Complexity vs. Threshold Change
        plt.figure(figsize=(10, 6))
        plt.scatter(scene_complexities, threshold_changes, alpha=0.7)
        plt.plot(np.unique(scene_complexities), 
                 np.poly1d(np.polyfit(scene_complexities, threshold_changes, 1))(np.unique(scene_complexities)), 
                 color='red')
        plt.xlabel('Scene Complexity')
        plt.ylabel('Threshold Change')
        plt.title('Scene Complexity vs. Threshold Adaptation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'complexity_vs_threshold.png'), dpi=300)
        plt.close()
        
        # --- NEW: Object Count Comparison (Scatter Plot) ---
        plt.figure(figsize=(10, 8))
        plt.scatter(standard_objects, adaptive_objects, alpha=0.5, s=10) # Smaller points for large dataset
        max_val = max(max(standard_objects) if standard_objects else 0, max(adaptive_objects) if adaptive_objects else 0)
        plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y=x (Equal Counts)')
        plt.xlabel('Standard Detection Object Count')
        plt.ylabel('Adaptive Detection Object Count')
        plt.title('Object Count Comparison (Standard vs. Adaptive)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal') # Ensure aspect ratio is equal
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'object_count_scatter.png'), dpi=300)
        plt.close()

        # --- NEW: Object Count Difference Distribution ---
        object_count_diff = np.array(adaptive_objects) - np.array(standard_objects)
        plt.figure(figsize=(10, 6))
        sns.histplot(object_count_diff, kde=True, bins=max(20, int(np.ptp(object_count_diff)) // 2) if len(object_count_diff)>0 else 10) # Auto-adjust bins somewhat
        mean_diff = np.mean(object_count_diff) if len(object_count_diff) > 0 else 0
        plt.axvline(x=mean_diff, color='r', linestyle='--', label=f'Mean Diff: {mean_diff:.2f}')
        plt.xlabel('Object Count Difference (Adaptive - Standard)')
        plt.ylabel('Frequency / Density')
        plt.title('Distribution of Object Count Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'object_count_difference_distribution.png'), dpi=300)
        plt.close()
        
        # --- NEW: Processing Time Comparison (Scatter Plot) ---
        plt.figure(figsize=(10, 8))
        plt.scatter(standard_times, adaptive_times, alpha=0.5, s=10)
        max_time = max(max(standard_times) if standard_times else 0, max(adaptive_times) if adaptive_times else 0)
        plt.plot([0, max_time], [0, max_time], color='red', linestyle='--', label='y=x (Equal Times)')
        plt.xlabel('Standard Detection Time (s)')
        plt.ylabel('Adaptive Detection Time (s)')
        plt.title('Processing Time Comparison (Standard vs. Adaptive)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.axis('equal')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'processing_time_scatter.png'), dpi=300)
        plt.close()

        # --- NEW: Processing Time Difference Distribution ---
        time_diff = np.array(standard_times) - np.array(adaptive_times) # Standard - Adaptive
        plt.figure(figsize=(10, 6))
        sns.histplot(time_diff, kde=True, bins=30)
        mean_time_diff = np.mean(time_diff) if len(time_diff) > 0 else 0
        plt.axvline(x=mean_time_diff, color='r', linestyle='--', label=f'Mean Time Saved: {mean_time_diff:.4f}s')
        plt.xlabel('Processing Time Difference (Standard - Adaptive) (s)')
        plt.ylabel('Frequency / Density')
        plt.title('Distribution of Processing Time Differences')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'processing_time_difference_distribution.png'), dpi=300)
        plt.close()
        
        # 4. Speed Improvement Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(speed_improvements, bins=10, kde=True)
        plt.axvline(x=np.mean(speed_improvements), color='r', linestyle='--', label=f'Mean: {np.mean(speed_improvements):.2f}x')
        plt.xlabel('Speed Improvement Factor')
        plt.ylabel('Frequency')
        plt.title('Distribution of Speed Improvements')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'speed_improvement_distribution.png'), dpi=300)
        plt.close()
        
        # 5. Threshold Change Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(threshold_changes, bins=10, kde=True)
        plt.axvline(x=np.mean(threshold_changes), color='r', linestyle='--', label=f'Mean: {np.mean(threshold_changes):.3f}')
        plt.xlabel('Threshold Change')
        plt.ylabel('Frequency')
        plt.title('Distribution of Threshold Changes')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'threshold_change_distribution.png'), dpi=300)
        plt.close()
        
        # 6. Complexity Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(scene_complexities, bins=10, kde=True)
        plt.axvline(x=np.mean(scene_complexities), color='r', linestyle='--', label=f'Mean: {np.mean(scene_complexities):.2f}')
        plt.xlabel('Scene Complexity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Scene Complexities')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'complexity_distribution.png'), dpi=300)
        plt.close()
        
        # 7. Object Count Change Scatter Plot
        object_count_diff = np.array(adaptive_objects) - np.array(standard_objects)
        plt.figure(figsize=(10, 6))
        plt.scatter(scene_complexities, object_count_diff, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Scene Complexity')
        plt.ylabel('Object Count Difference (Adaptive - Standard)')
        plt.title('Impact of Scene Complexity on Object Detection')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'complexity_vs_detection_diff.png'), dpi=300)
        plt.close()
        
    # Generate markdown report
    with open(os.path.join(output_dir, "experiment_report.md"), "w") as f:
        f.write("# AdaptiVision Experiment Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Total images processed: {len(results)}\n")
        f.write(f"- Successfully processed: {success_count}\n")
        f.write(f"- Failed: {failure_count}\n\n")
        
        if success_count > 0:
            f.write("## Key Findings\n\n")
            f.write(f"- Average scene complexity: {summary['avg_scene_complexity']:.3f}\n")
            f.write(f"- Average threshold adjustment: {summary['avg_threshold_change']:.3f}\n")
            f.write(f"- Average objects detected (standard): {summary['avg_standard_objects']:.2f}\n")
            f.write(f"- Average objects detected (adaptive): {summary['avg_adaptive_objects']:.2f}\n")
            f.write(f"- Average processing time (standard): {summary['avg_standard_time']:.4f}s\n")
            f.write(f"- Average processing time (adaptive): {summary['avg_adaptive_time']:.4f}s\n")
            f.write(f"- Average speed improvement: {summary['avg_speed_improvement']:.2f}x\n\n")
            
            f.write("## Visualizations\n\n")
            f.write("### Scene Complexity vs. Threshold Adaptation\n\n")
            f.write("![Complexity vs Threshold](analytics/complexity_vs_threshold.png)\n\n")
            
            f.write("### Object Detection Comparison\n\n")
            f.write("This scatter plot compares the number of objects detected by the standard method (x-axis) versus the adaptive method (y-axis) for each image. Points on the red dashed line indicate both methods found the same number of objects.\n\n")
            f.write("![Object Count Scatter](analytics/object_count_scatter.png)\n\n")
            f.write("This histogram shows the distribution of the difference in object counts (Adaptive - Standard). Positive values indicate the adaptive method found more objects.\n\n")
            f.write("![Object Count Difference Distribution](analytics/object_count_difference_distribution.png)\n\n")
            
            f.write("### Processing Time Comparison\n\n")
            f.write("This scatter plot compares the processing time per image for the standard method (x-axis) versus the adaptive method (y-axis). Points on the red dashed line indicate equal processing time.\n\n")
            f.write("![Processing Time Scatter](analytics/processing_time_scatter.png)\n\n")
            f.write("This histogram shows the distribution of the difference in processing time (Standard - Adaptive). Positive values indicate the adaptive method was faster.\n\n")
            f.write("![Processing Time Difference Distribution](analytics/processing_time_difference_distribution.png)\n\n")
            
            f.write("### Speed Improvement Distribution\n\n")
            f.write("This histogram shows the distribution of the speed improvement factor (Standard Time / Adaptive Time).\n\n")
            
        f.write("## Sample Comparisons\n\n")
        
        # Add a few sample comparisons
        success_results = [r for r in results if r["comparison"].get("success", False)]
        for i, img_data in enumerate(success_results[:5]):
            if i >= len(success_results):
                break
                
            filename = img_data["filename"]
            comparison_path = img_data["comparison"]["output_path"]
            rel_path = os.path.relpath(comparison_path, output_dir)
            
            f.write(f"### Sample {i+1}: {filename}\n\n")
            f.write(f"![Comparison]({rel_path})\n\n")
            f.write("**Details:**\n\n")
            f.write(f"- Scene complexity: {img_data['adaptive_detection']['scene_complexity']:.3f}\n")
            f.write(f"- Adaptive threshold: {img_data['adaptive_detection']['adaptive_threshold']:.3f}\n")
            f.write(f"- Standard detection: {img_data['standard_detection']['object_count']} objects\n")
            f.write(f"- Adaptive detection: {img_data['adaptive_detection']['object_count']} objects\n")
            f.write(f"- Speed improvement: {img_data['comparison']['speed_improvement']:.2f}x\n\n")
    
    print(f"Analytics and report generated in {output_dir}")

def main():
    args = parse_args()
    # Add a flag to trigger reanalysis if needed (can be set via another arg later if desired)
    # For now, let's assume we always process if calling main directly without a specific reanalyze flag
    run_experiments(
        data_dir=args.data,
        output_dir=args.output,
        model_path=args.weights,
        device=args.device,
        reanalyze_only=False # Set to True manually or via arg if needed
    )

if __name__ == "__main__":
    main() 