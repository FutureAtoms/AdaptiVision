import argparse
import json
import os # Added os import
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np # Added numpy import
import matplotlib.pyplot as plt # Added plt import

# --- ADDITION: Import Ultralytics components ---
from pathlib import Path
from ultralytics.utils.checks import check_dataset
from ultralytics.settings import Settings, SETTINGS

# --- ADDITION: Function to check and download COCO ---
# (Identical to the one added in run_experiments.py)
def check_and_download_coco(annotation_file, dataset_name="coco"):
    """Checks if the specified dataset (e.g., COCO) exists locally and downloads it if not."""
    is_coco = dataset_name.lower() == "coco"
    
    # Infer expected base directory from annotation file path
    try:
        ann_path = Path(annotation_file)
        expected_coco_dir = ann_path.parent.parent # e.g., ./datasets/coco/annotations/.. /.. = ./datasets/coco
        expected_img_dir = expected_coco_dir / "images" / "val2017"
        expected_ann_file = expected_coco_dir / "annotations" / "instances_val2017.json"
        
        # Double check the inference makes sense
        if not str(annotation_file).endswith("instances_val2017.json") or expected_coco_dir.name != 'coco':
             print(f"Warning: Annotation file path '{annotation_file}' doesn't match expected COCO structure. Skipping download check.")
             return
             
    except Exception as path_e:
        print(f"Warning: Could not reliably determine expected COCO directory from '{annotation_file}': {path_e}. Skipping download check.")
        return

    # Only proceed if using COCO
    if not is_coco:
        print(f"Skipping automatic download check for non-COCO dataset: {dataset_name}")
        return

    if expected_img_dir.exists() and expected_ann_file.exists():
        print(f"Found existing {dataset_name} dataset at: {expected_coco_dir}")
        return
    else:
        print(f"{dataset_name} dataset not found or incomplete at {expected_coco_dir}. Attempting download...")
        
        # Ensure the root datasets directory exists
        expected_coco_dir.parent.mkdir(parents=True, exist_ok=True)
        
        # Update Ultralytics settings to download to ./datasets
        original_settings = SETTINGS.copy()
        try:
            print(f"Updating Ultralytics datasets_dir to: {expected_coco_dir.parent.resolve()}")
            SETTINGS.update({'datasets_dir': str(expected_coco_dir.parent.resolve())})
            
            # Trigger download using check_dataset
            print(f"Triggering download for {dataset_name}...")
            data_info = check_dataset(dataset_name + ".yaml") # Use the YAML name recognized by ultralytics
            
            # Verify download location
            if not expected_img_dir.exists() or not expected_ann_file.exists():
                 print(f"Warning: Download finished, but expected files/dirs still not found in {expected_coco_dir}. Please check download location or manually place the dataset.")
                 print(f"Ultralytics check_dataset returned: {data_info}")
            else:
                 print(f"Successfully downloaded and verified {dataset_name} dataset at {expected_coco_dir}")

        except Exception as e:
            print(f"Error during {dataset_name} download: {e}")
            print(f"Please ensure you have internet connectivity and necessary permissions.")
            print(f"You may need to download the COCO dataset manually and place it in {expected_coco_dir}")
        finally:
            # Restore original settings
            SETTINGS.update(original_settings)
            print(f"Restored original Ultralytics settings.")


def run_evaluation(coco_gt, coco_dt, iou_type, img_ids=None, results_file_path=None):
    """Runs COCO evaluation for a given set of image IDs."""
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    
    if img_ids is not None:
        print(f"Evaluating on {len(img_ids)} images...")
        coco_eval.params.imgIds = img_ids
    else:
        print(f"Evaluating on all images...")

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize() # Prints the standard COCO metrics
    
    # --- ADD: Calculate and Print Per-Class AP --- 
    print("\nPer-Class AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]:")
    # Precision array shape: (iou_thresholds, recall_thresholds, classes, areas, max_detections)
    # We want average precision over IoU thresholds for each class (K)
    # using area=all (index 0) and maxDets=100 (index 2)
    precision = coco_eval.eval['precision']
    num_classes = precision.shape[2]
    class_names = coco_gt.loadCats(coco_gt.getCatIds())
    class_name_map = {cat['id']: cat['name'] for cat in class_names}
    
    results_per_category = []
    for k_idx in range(num_classes):
        category_id = coco_gt.getCatIds()[k_idx] # Get the actual category ID
        category_name = class_name_map.get(category_id, f"ClassID_{category_id}")
        
        # Get precision for this class across all IoU thresholds and recall thresholds
        # slice: [IoUs, Recalls, Class k_idx, Area all, MaxDets 100]
        s = precision[:, :, k_idx, 0, 2]
        
        if len(s[s > -1]) == 0:
            ap = -1 # Assign -1 if no valid precision values (e.g., class not present)
        else:
            ap = np.mean(s[s > -1])
            
        results_per_category.append((category_name, f'{ap:.3f}'))
        
    # Sort by class name for consistent output
    results_per_category.sort(key=lambda x: x[0])
    
    # Print in columns for better readability
    num_cols = 3
    col_width = 25
    for i in range(0, len(results_per_category), num_cols):
        line = ""
        for j in range(num_cols):
            if i + j < len(results_per_category):
                 item = results_per_category[i+j]
                 line += f"{item[0]:<18}: {item[1]:<5}" + " | "
        print(line.strip().strip('|').strip())
    # --- END ADD ---

    # --- ADD: Generate and Save PR Curve --- 
    if img_ids is None and results_file_path: # Only plot overall curve for now
        try:
            # Precision shape: (iou_thresholds, recall_thresholds, classes, areas, max_detections)
            # Recall thresholds: coco_eval.params.recThrs
            precision = coco_eval.eval['precision']
            recall = coco_eval.params.recThrs
            
            # Get precision at IoU=0.50 (first IoU threshold index 0)
            # Averaged across all classes (mean over K), area=all (index 0), maxDets=100 (index 2)
            pr_iou50 = precision[0, :, :, 0, 2].mean(axis=1)
            
            plt.figure()
            plt.plot(recall, pr_iou50, marker='.')
            plt.xlabel('Recall')
            plt.ylabel('Precision @ IoU=0.50')
            plt.title('Precision-Recall Curve (All Classes, IoU=0.50)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.grid(True)
            
            pr_curve_path = results_file_path.replace('_preds.json', '_pr_curve.png')
            plt.savefig(pr_curve_path)
            print(f"\nPR curve saved to: {pr_curve_path}")
            plt.close()
        except Exception as e:
            print(f"\nWarning: Failed to generate PR curve: {e}")
    # --- END ADD ---

    return coco_eval.stats # Return the stats array

def main(args):
    # --- ADDITION: Check and download dataset ---
    # Assuming dataset name is 'coco' if the default annotation file name is used
    dataset_name = "coco" if "instances_val2017.json" in args.annotation_file else None
    if dataset_name:
         check_and_download_coco(args.annotation_file, dataset_name)
    # --- END ADDITION ---
    
    # Ensure annotation file exists before proceeding
    if not os.path.exists(args.annotation_file):
         print(f"Error: Ground truth annotation file not found after check: {args.annotation_file}")
         print("Please ensure the dataset was downloaded correctly or provide the correct path.")
         return

    coco_gt = COCO(args.annotation_file)

    # Load results
    coco_dt = None
    try:
        # Load results using loadRes which handles list or file path
        coco_dt = coco_gt.loadRes(args.results_file)
        if not coco_dt.anns:
             print(f"Warning: Prediction file '{args.results_file}' loaded but contains no annotations. Skipping evaluation.")
             coco_dt = None
    except (json.JSONDecodeError, FileNotFoundError, TypeError) as e:
         print(f"Error loading prediction file '{args.results_file}': {e}. Skipping evaluation.")
         return # Exit if predictions cannot be loaded

    if not coco_dt:
        return

    # --- Overall Evaluation --- 
    print(f"\n--- Running COCO Evaluation (Overall) for: {args.results_file} ---")
    overall_stats = run_evaluation(coco_gt, coco_dt, args.iou_type, results_file_path=args.results_file)
    print("--- Overall Evaluation Complete ---")

    # --- Complexity-Based Evaluation --- 
    complexity_file_path = args.results_file.replace('_preds.json', '_complexity.json')
    if os.path.exists(complexity_file_path):
        print(f"\nFound complexity file: {complexity_file_path}")
        with open(complexity_file_path, 'r') as f:
            complexity_data = json.load(f)
        
        # Convert keys (image_ids) to integers if they are strings
        complexity_data = {int(k): v for k, v in complexity_data.items()}

        # Get all image IDs present in the ground truth
        all_gt_img_ids = coco_gt.getImgIds()

        # Define complexity bins and get image IDs for each
        # Bins based on AdaptiVision._get_adaptive_threshold logic
        bins = {
            "Low Complexity (< 0.3)": [],
            "Medium Complexity (0.3-0.7)": [],
            "High Complexity (> 0.7)": []
        }

        missing_complexity_count = 0
        for img_id in all_gt_img_ids:
            complexity_score = complexity_data.get(img_id, None) 
            if complexity_score is None:
                # Assign to medium if complexity score is missing for a GT image?
                # Or skip? Let's skip for now and report.
                missing_complexity_count += 1
                continue 

            if complexity_score < 0.3:
                bins["Low Complexity (< 0.3)"].append(img_id)
            elif complexity_score <= 0.7: # Include 0.7 in medium
                bins["Medium Complexity (0.3-0.7)"].append(img_id)
            else:
                bins["High Complexity (> 0.7)"].append(img_id)
        
        if missing_complexity_count > 0:
             print(f"Warning: Could not find complexity scores for {missing_complexity_count} images present in ground truth.")

        # Run evaluation for each bin
        print("\n--- Running Complexity-Based Evaluation ---")
        for bin_name, img_ids in bins.items():
            print(f"\n--- Evaluating Bin: {bin_name} ({len(img_ids)} images) ---")
            if not img_ids:
                print("No images in this bin. Skipping evaluation.")
                continue
            
            # It's important to use the SAME coco_dt object, 
            # but restrict the evaluation to specific img_ids via params
            # Don't pass results_file_path here to avoid plotting PR curve for each bin
            run_evaluation(coco_gt, coco_dt, args.iou_type, img_ids=img_ids)
            print(f"--- Bin Evaluation Complete: {bin_name} ---")
        
    else:
        print(f"\nComplexity file not found at {complexity_file_path}. Skipping complexity-based evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCO format predictions using pycocotools, optionally with complexity breakdown.")
    parser.add_argument('--annotation-file', type=str, required=True, help="Path to the COCO ground truth JSON file (e.g., ./datasets/coco/annotations/instances_val2017.json).")
    parser.add_argument('--results-file', type=str, required=True, help="Path to the prediction results JSON file (e.g., *_preds.json).")
    parser.add_argument('--iou-type', type=str, default='bbox', choices=['bbox', 'segm', 'keypoints'], help="Type of evaluation.")

    args = parser.parse_args()
    main(args) 