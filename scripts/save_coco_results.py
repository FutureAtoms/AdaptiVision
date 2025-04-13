import argparse
import json
import os
from pathlib import Path
import sys # Add sys import

# --- ADD PATH FOR SRC IMPORT --- 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# --- END PATH ADDITION ---

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywhn # For bbox conversion

# Import the main AdaptiVision class
from src.adaptivision import AdaptiVision

# COCO128 class indices to official COCO category IDs mapping
# This might need adjustment based on your specific COCO128 setup/annotations
# You can often get this mapping from the dataset's YAML file or annotation file
# Example mapping (VERIFY THIS IS CORRECT FOR COCO128 as used by YOLOv8):
COCO80_TO_COCO91_MAP = { 0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
                       16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33,
                       29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47,
                       42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60,
                       55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77,
                       68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90 }

def get_image_id(image_path, coco_gt): 
    """Helper to find COCO image ID from filename."""
    filename = os.path.basename(image_path)
    for img_id, img_info in coco_gt.imgs.items():
        if img_info['file_name'] == filename:
            return img_id
    print(f"Warning: Could not find image_id for {filename}")
    return None # Or raise an error

def run_predictions(args):
    coco_results = []
    # --- ADD: Store per-image complexity --- 
    complexity_results = {}
    # --- END ADD ---
    image_files = sorted(list(Path(args.dataset_path).rglob('*.jpg')))
    
    # Load ground truth to get image IDs (needed for COCO format)
    # This requires the annotation file to be available
    gt_annotation_file = args.gt_annotations
    if not os.path.exists(gt_annotation_file):
        print(f"Error: Ground truth annotation file not found at {gt_annotation_file}")
        print("Please provide the correct path using --gt-annotations")
        print("This is needed to map filenames to image_ids for COCO results format.")
        return
        
    from pycocotools.coco import COCO # Import here to avoid dependency if file missing
    coco_gt = COCO(gt_annotation_file)

    print("Initializing Base YOLO Model...")
    base_model = YOLO(args.weights)
    
    adaptivision_instance = None
    if args.method == 'adaptivision':
        print("Initializing AdaptiVision Settings...")
        # Instantiate AdaptiVision class - it loads its own model internally, but we use the base_model for initial predict
        adaptivision_instance = AdaptiVision(model_path=args.weights, # Needs path for internal settings if any
                                           device=args.device, # Pass the device argument
                                           enable_adaptive_confidence=args.adaptive, # Use arg flag
                                           context_aware=args.context,                # Use arg flag
                                           enable_postprocess_filter=args.postprocess_filter) # ADD FLAG
                                           
    initial_conf_threshold = 0.05 # Use a low threshold for the initial pass

    print(f"Processing {len(image_files)} images from {args.dataset_path}...")
    for img_path in tqdm(image_files, desc=f"Predicting - {args.method}"):
        img_path_str = str(img_path)
        image_id = get_image_id(img_path_str, coco_gt)
        if image_id is None:
            continue # Skip if image_id not found
            
        # --- Perform Initial Low-Confidence Prediction --- 
        try:
            # Always run base model with low confidence first
            initial_results = base_model(img_path_str, verbose=False, conf=initial_conf_threshold)
        except Exception as e:
             print(f"\nError during Initial YOLO prediction for {img_path_str}: {e}")
             continue

        # --- Extract Initial Detections --- 
        final_boxes_for_coco = []
        final_scores_for_coco = []
        final_classes_for_coco = []
        img_h, img_w = (0, 0)
        processed_results = False # Flag to track if results were processed

        if initial_results and len(initial_results[0].boxes) > 0:
            res = initial_results[0] # Results for the first (only) image
            initial_boxes = res.boxes.xyxy.cpu().numpy() # Use xyxy for AdaptiVision internal processing
            initial_scores = res.boxes.conf.cpu().numpy() 
            initial_classes = res.boxes.cls.cpu().numpy().astype(int)
            initial_class_names = [res.names[int(cls_id)] for cls_id in initial_classes]
            img_h, img_w = res.orig_shape

            # --- ADD: Calculate Scene Complexity (if adaptivision) --- 
            scene_complexity = -1 # Default/invalid value
            if args.method == 'adaptivision' and adaptivision_instance:
                # Use the initial low-confidence detections for complexity calculation
                try:
                    scene_complexity = adaptivision_instance._calculate_scene_complexity(
                        initial_boxes, initial_class_names, (img_h, img_w, 3) # Assuming 3 channels
                    )
                    # Store complexity keyed by image_id
                    complexity_results[int(image_id)] = float(scene_complexity) # Convert to standard float
                except Exception as e:
                    print(f"\nWarning: Could not calculate scene complexity for image {image_id}: {e}")
            # --- END ADD ---

            # --- Apply Filtering Method --- 
            if args.method == 'adaptivision':
                if adaptivision_instance:
                    try:
                        # Call AdaptiVision's internal post-processing 
                        # Pass initial detections to it
                        # Assuming _post_process_detections returns filtered results
                        filtered_boxes, filtered_scores, filtered_classes, filtered_names = adaptivision_instance._post_process_detections(
                            initial_boxes, initial_scores, initial_classes, initial_class_names, res.orig_shape
                        )
                        # IMPORTANT: We use the filtered boxes/classes, but need the ORIGINAL scores associated with them
                        # Need to map filtered results back to original scores. This assumes order might be preserved or requires matching.
                        # Let's assume for now _post_process_detections *could* return indices or a way to map back.
                        # SIMPLIFICATION FOR NOW: Use the scores that came out of _post_process_detections, 
                        # acknowledging this might not be purely the original score if AdaptiVision modifies them internally.
                        # A more robust implementation would modify _post_process_detections.
                        final_boxes_for_coco = filtered_boxes # These are xyxy
                        final_scores_for_coco = filtered_scores # Use scores returned by post-processing
                        final_classes_for_coco = filtered_classes
                        processed_results = True
                    except Exception as e:
                         print(f"\nError during AdaptiVision post-processing for {img_path_str}: {e}")
                else:
                     print("\nError: AdaptiVision instance not created.")
                     
            elif args.method == 'baseline':
                 # Baseline: Filter initial results manually by target threshold (e.g., 0.25 or the --baseline-conf for mAP)
                 # FOR mAP: We actually want *all* results above the initial_conf_threshold, 
                 # because the COCO eval tool applies the score thresholding. So, we use the initial results directly.
                 # If you wanted to replicate the fixed 0.25 threshold count, you would filter here by score > 0.25. 
                 
                 baseline_map_conf = args.baseline_conf # Use the low threshold for mAP
                 indices_to_keep = [i for i, score in enumerate(initial_scores) if score >= baseline_map_conf]
                 
                 final_boxes_for_coco = initial_boxes[indices_to_keep] # xyxy
                 final_scores_for_coco = initial_scores[indices_to_keep]
                 final_classes_for_coco = initial_classes[indices_to_keep]
                 processed_results = True

        # --- Process Final Detections into COCO Format --- 
        if processed_results and len(final_boxes_for_coco) > 0:
            # Convert final xyxy boxes to xywh for COCO format
            # Need absolute coordinates, not normalized for this conversion typically
            # boxes_xywh = xyxy2xywh(final_boxes_for_coco) # Requires torch tensor usually
            
            # Manual conversion from xyxy (absolute) to xywh (absolute)
            final_boxes_xywh = []
            for box_xyxy in final_boxes_for_coco:
                x1, y1, x2, y2 = box_xyxy
                w = x2 - x1
                h = y2 - y1
                final_boxes_xywh.append([x1, y1, w, h])
            final_boxes_xywh = np.array(final_boxes_xywh)
            
            if len(final_boxes_xywh) > 0: # Check if any boxes remain after conversion
                for i in range(len(final_boxes_xywh)):
                    class_id_yolo = final_classes_for_coco[i]
                    category_id_coco = COCO80_TO_COCO91_MAP.get(class_id_yolo, -1)
                    
                    if category_id_coco == -1:
                        # print(f"Warning: Skipping detection with unknown YOLO class index {class_id_yolo} for {img_path_str}")
                        continue
                        
                    abs_x, abs_y, abs_w, abs_h = final_boxes_xywh[i]
                        
                    coco_results.append({
                        "image_id": int(image_id), # Ensure image_id is standard int
                        "category_id": int(category_id_coco), # Ensure category_id is standard int
                        "bbox": [
                            round(float(abs_x), 2), 
                            round(float(abs_y), 2), 
                            round(float(abs_w), 2), 
                            round(float(abs_h), 2)
                        ],
                        "score": round(float(final_scores_for_coco[i]), 4) # Cast score to float
                    })

    # Save results to JSON
    print(f"Saving {len(coco_results)} detections to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(coco_results, f, indent=2)
    print("Done.")

    # --- ADD: Save Complexity Results --- 
    if args.method == 'adaptivision' and complexity_results:
        complexity_output_path = args.output_json.replace('_preds.json', '_complexity.json')
        print(f"Saving {len(complexity_results)} complexity scores to {complexity_output_path}...")
        with open(complexity_output_path, 'w') as f:
            json.dump(complexity_results, f, indent=2)
        print("Complexity scores saved.")
    # --- END ADD ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate COCO format predictions for mAP evaluation.")
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset images directory (e.g., datasets/coco128/images/train2017).")
    parser.add_argument('--gt-annotations', type=str, required=True, help="Path to the COCO ground truth JSON file (e.g., datasets/coco128/annotations/instances_val2017.json).")
    parser.add_argument('--weights', type=str, default='weights/model_n.pt', help="Path to YOLOv8 model weights (.pt file).")
    parser.add_argument('--output-json', type=str, required=True, help="Path to save the output COCO results JSON file.")
    parser.add_argument('--method', type=str, required=True, choices=['baseline', 'adaptivision'], help="Method to run ('baseline' or 'adaptivision').")
    parser.add_argument('--device', type=str, default='auto', help="Device to run inference on ('auto', 'cpu', 'cuda', 'mps').")
    parser.add_argument('--baseline-conf', type=float, default=0.001, help="Confidence threshold for baseline YOLO predict for mAP eval (use low value). Not used by AdaptiVision mode.")
    # --- ADD ABLATION FLAGS ---
    parser.add_argument('--adaptive', action=argparse.BooleanOptionalAction, default=True, help="Enable adaptive confidence thresholding (only for --method adaptivision).")
    parser.add_argument('--context', action=argparse.BooleanOptionalAction, default=True, help="Enable context-aware reasoning (only for --method adaptivision).")
    parser.add_argument('--postprocess-filter', action=argparse.BooleanOptionalAction, default=True, help="Enable geometric post-processing filter (only for --method adaptivision).") # ADD FLAG
    # --- END ABLATION FLAGS ---

    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    
    run_predictions(args) 