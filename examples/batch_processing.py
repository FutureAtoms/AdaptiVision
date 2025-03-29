#!/usr/bin/env python
"""
Example script for batch processing multiple images with AdaptiVision.
"""
import os
import argparse
import time
import glob
import cv2
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add the parent directory to the path to import the AdaptiVision module
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.adaptivision import AdaptiVision

def parse_args():
    parser = argparse.ArgumentParser(description="Batch process images with AdaptiVision")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input images")
    parser.add_argument("--output-dir", type=str, default="results/batch", help="Directory for output images")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Base confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto, cpu, cuda, mps)")
    parser.add_argument("--disable-adaptive", action="store_true", help="Disable adaptive confidence")
    parser.add_argument("--disable-context", action="store_true", help="Disable context-aware reasoning")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (0 for sequential)")
    parser.add_argument("--image-types", type=str, default="jpg,jpeg,png", help="Comma-separated list of image extensions")
    parser.add_argument("--save-json", action="store_true", help="Save detection results as JSON")
    return parser.parse_args()

def process_image(input_path, output_dir, detector_config, save_json=False):
    """Process a single image and save results"""
    try:
        # Get output filename
        filename = os.path.basename(input_path)
        output_path = os.path.join(output_dir, filename)
        
        # Initialize detector (creates a new instance for each process)
        detector = AdaptiVision(**detector_config)
        
        # Run detection
        start_time = time.time()
        results = detector.predict(input_path)
        inference_time = time.time() - start_time
        
        # Skip if no results
        if not results or len(results) == 0:
            return {
                "image": input_path,
                "output": None,
                "inference_time": inference_time,
                "objects_detected": 0,
                "success": False,
                "error": "No detections found"
            }
        
        # Save visualization
        detector.visualize(input_path, results[0], output_path)
        
        # Prepare result summary
        result_summary = {
            "image": input_path,
            "output": output_path,
            "inference_time": inference_time,
            "objects_detected": len(results[0]['boxes']),
            "success": True
        }
        
        # Add adaptive threshold info if available
        if detector.enable_adaptive_confidence and 'adaptive_threshold' in results[0]:
            result_summary["complexity"] = results[0]['scene_complexity']
            result_summary["adaptive_threshold"] = results[0]['adaptive_threshold']
            result_summary["base_threshold"] = detector_config["conf_threshold"]
        
        # Save detailed JSON results if requested
        if save_json:
            json_path = os.path.splitext(output_path)[0] + '.json'
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = []
            for result in results:
                json_result = {}
                for key, value in result.items():
                    if key == 'boxes':
                        json_result[key] = value.tolist() if hasattr(value, 'tolist') else value
                    elif key == 'scores':
                        json_result[key] = value.tolist() if hasattr(value, 'tolist') else value
                    elif key == 'labels':
                        json_result[key] = value.tolist() if hasattr(value, 'tolist') else value
                    else:
                        json_result[key] = value
                json_results.append(json_result)
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            result_summary["json_output"] = json_path
        
        return result_summary
        
    except Exception as e:
        return {
            "image": input_path,
            "output": None,
            "success": False,
            "error": str(e)
        }

def main():
    # Parse arguments
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = args.image_types.lower().split(',')
    image_files = []
    for ext in image_extensions:
        ext = ext.strip()
        if not ext.startswith('.'):
            ext = '.' + ext
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(args.input_dir, f"*{ext.upper()}")))
    
    # Remove duplicates and sort
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"No images found in {args.input_dir} with extensions: {args.image_types}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create detector configuration (shared between processes)
    detector_config = {
        "model_path": args.weights,
        "device": args.device,
        "conf_threshold": args.conf_thres,
        "iou_threshold": args.iou_thres,
        "enable_adaptive_confidence": not args.disable_adaptive,
        "context_aware": not args.disable_context
    }
    
    # Process images
    results = []
    total_start_time = time.time()
    
    if args.workers <= 0:
        # Sequential processing
        print("Processing images sequentially...")
        for input_path in tqdm(image_files):
            result = process_image(input_path, args.output_dir, detector_config, args.save_json)
            results.append(result)
    else:
        # Parallel processing
        print(f"Processing images in parallel with {args.workers} workers...")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            futures = [
                executor.submit(process_image, input_path, args.output_dir, detector_config, args.save_json)
                for input_path in image_files
            ]
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
    
    total_time = time.time() - total_start_time
    
    # Compute statistics
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    if successful:
        total_objects = sum(r.get('objects_detected', 0) for r in successful)
        avg_objects = total_objects / len(successful)
        avg_time = sum(r.get('inference_time', 0) for r in successful) / len(successful)
        
        # Compute adaptive threshold statistics if available
        adaptive_results = [r for r in successful if 'adaptive_threshold' in r]
        if adaptive_results:
            avg_complexity = sum(r.get('complexity', 0) for r in adaptive_results) / len(adaptive_results)
            avg_threshold = sum(r.get('adaptive_threshold', 0) for r in adaptive_results) / len(adaptive_results)
            base_threshold = adaptive_results[0].get('base_threshold', args.conf_thres)
            
            # Count threshold adjustments
            decreased = sum(1 for r in adaptive_results if r.get('adaptive_threshold', 0) < base_threshold)
            increased = sum(1 for r in adaptive_results if r.get('adaptive_threshold', 0) > base_threshold)
            unchanged = len(adaptive_results) - decreased - increased
    
    # Print summary
    print("\n" + "="*50)
    print(f"Batch Processing Summary:")
    print(f"  - Total images: {len(image_files)}")
    print(f"  - Successfully processed: {len(successful)}")
    print(f"  - Failed: {len(failed)}")
    print(f"  - Total processing time: {total_time:.2f} seconds")
    
    if successful:
        print(f"\nDetection Statistics:")
        print(f"  - Total objects detected: {total_objects}")
        print(f"  - Average objects per image: {avg_objects:.2f}")
        print(f"  - Average inference time: {avg_time*1000:.2f} ms per image")
        
        if adaptive_results:
            print(f"\nAdaptive Threshold Statistics:")
            print(f"  - Average scene complexity: {avg_complexity:.3f}")
            print(f"  - Average adaptive threshold: {avg_threshold:.3f}")
            print(f"  - Base threshold: {base_threshold:.3f}")
            print(f"  - Threshold adjustments:")
            print(f"      Decreased: {decreased} images")
            print(f"      Increased: {increased} images")
            print(f"      Unchanged: {unchanged} images")
    
    if failed:
        print(f"\nFailed Images:")
        for i, fail in enumerate(failed[:5]):  # Show only first 5 failures
            print(f"  {i+1}. {os.path.basename(fail['image'])}: {fail.get('error', 'Unknown error')}")
        
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print("\nResults saved to:", args.output_dir)

if __name__ == "__main__":
    main() 