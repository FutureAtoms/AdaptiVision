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

# Add parent directory to path to import AdaptiVision modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import AdaptiVision modules
from src.adaptivision import AdaptiVision

def run_experiments(data_dir, output_dir, model_path="weights/model_n.pt"):
    """
    Run comprehensive experiments comparing standard YOLO with AdaptiVision
    
    Args:
        data_dir: Directory containing input images
        output_dir: Directory to save results
        model_path: Path to YOLO model weights
    """
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "standard"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "adaptive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "comparisons"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "analytics"), exist_ok=True)
    
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
    
    # Initialize detectors
    standard_detector = AdaptiVision(
        model_path=model_path,
        enable_adaptive_confidence=False,
        context_aware=False
    )
    
    adaptive_detector = AdaptiVision(
        model_path=model_path, 
        enable_adaptive_confidence=True,
        context_aware=True
    )
    
    # Prepare data collection
    results = []
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
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
            standard_results = standard_detector.predict(img_path)
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
                standard_detector.visualize(img_path, standard_results[0], standard_img_path)
                
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
            adaptive_results = adaptive_detector.predict(img_path)
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
                adaptive_detector.visualize(img_path, adaptive_results[0], adaptive_img_path)
                
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
                    from src.create_visualizations import create_complexity_visualization, create_threshold_map
                    
                    # Complexity visualization
                    complexity_path = os.path.join(visualization_dir, f"complexity_{img_filename}")
                    create_complexity_visualization(img_path, complexity_path, scene_complexity)
                    
                    # Threshold map
                    threshold_path = os.path.join(visualization_dir, f"threshold_map_{img_filename}")
                    create_threshold_map(img_path, threshold_path, scene_complexity, 0.25, adaptive_threshold)
                    
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
                
                # Create comparison image
                create_comparison_image(
                    img_path,
                    standard_result,
                    adaptive_result,
                    comparison_path
                )
                
                # Add comparison info
                base_threshold = 0.25
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
        
        # 2. Object Count Comparison
        plt.figure(figsize=(12, 6))
        indices = range(len(filenames))
        bar_width = 0.35
        
        plt.bar(indices, standard_objects, bar_width, alpha=0.7, label='Standard Detection')
        plt.bar([i + bar_width for i in indices], adaptive_objects, bar_width, alpha=0.7, label='Adaptive Detection')
        
        plt.xlabel('Images')
        plt.ylabel('Object Count')
        plt.title('Standard vs. Adaptive Detection Object Count')
        plt.xticks([i + bar_width/2 for i in indices], [f.split('.')[0] for f in filenames], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'object_count_comparison.png'), dpi=300)
        plt.close()
        
        # 3. Processing Time Comparison
        plt.figure(figsize=(12, 6))
        
        plt.bar(indices, standard_times, bar_width, alpha=0.7, label='Standard Detection')
        plt.bar([i + bar_width for i in indices], adaptive_times, bar_width, alpha=0.7, label='Adaptive Detection')
        
        plt.xlabel('Images')
        plt.ylabel('Processing Time (s)')
        plt.title('Processing Time Comparison')
        plt.xticks([i + bar_width/2 for i in indices], [f.split('.')[0] for f in filenames], rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(analytics_dir, 'processing_time_comparison.png'), dpi=300)
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
            f.write("![Object Count](analytics/object_count_comparison.png)\n\n")
            
            f.write("### Processing Time Comparison\n\n")
            f.write("![Processing Time](analytics/processing_time_comparison.png)\n\n")
            
            f.write("### Speed Improvement Distribution\n\n")
            f.write("![Speed Improvement](analytics/speed_improvement_distribution.png)\n\n")
            
            f.write("### Threshold Change Distribution\n\n")
            f.write("![Threshold Change](analytics/threshold_change_distribution.png)\n\n")
            
            f.write("### Scene Complexity Distribution\n\n")
            f.write("![Complexity Distribution](analytics/complexity_distribution.png)\n\n")
            
            f.write("### Impact of Scene Complexity on Detection\n\n")
            f.write("![Complexity vs Detection Difference](analytics/complexity_vs_detection_diff.png)\n\n")
            
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
    parser = argparse.ArgumentParser(description="Run comprehensive experiments")
    parser.add_argument("--data", type=str, required=True, help="Directory with input images")
    parser.add_argument("--output", type=str, required=True, help="Directory for output")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    
    args = parser.parse_args()
    run_experiments(args.data, args.output, args.weights)

if __name__ == "__main__":
    main() 