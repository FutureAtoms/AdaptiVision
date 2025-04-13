# AdaptiVision Script Usage Guide

This document details the purpose and usage of the various Python scripts included in the AdaptiVision project.

## Quick Script Selection Guide

*   **For quick predictions and visual output on single images or directories:** Use `src/adaptivision.py`.
*   **For generating standard COCO evaluation metrics (like mAP):** Use the `scripts/save_coco_results.py` + `scripts/evaluate_coco.py` pipeline.
*   **For a detailed comparison between standard YOLO and AdaptiVision with ablation studies and analytics:** Use `scripts/run_experiments.py`.
*   **For generating specific plots from experiment results:** Use `scripts/generate_..._plot.py` scripts.

## Detailed Script Descriptions

### 1. `src/adaptivision.py` (Main Prediction Script)

*   **Purpose:** Performs object detection using either standard YOLO settings or the AdaptiVision enhancements on a single image or all images within a specified directory. Saves annotated output images.
*   **When to Use:** Ideal for quick testing, generating visual examples, or applying the model to a set of images when formal COCO evaluation is not the primary goal.
*   **Key Options:**
    *   `--input_path`: (Required) Path to a single input image file or a directory containing images.
    *   `--output_dir`: Directory where annotated output images will be saved (default: `results/output`).
    *   `--weights`: Path to the model weights file (default: `weights/model_n.pt`).
    *   `--device`: Processing device ('auto', 'cpu', 'cuda', 'mps') (default: 'auto').
    *   `--conf`: Base confidence threshold (default: 0.25).
    *   `--iou`: IoU threshold for Non-Maximum Suppression (NMS) (default: 0.45).
    *   `--adaptive`: Enable AdaptiVision's adaptive confidence thresholding (flag, default: disabled).
    *   `--context`: Enable AdaptiVision's context-aware reasoning (flag, default: disabled).
    *   `--classes`: Filter detections by specific class IDs (e.g., `--classes 0 2` for persons and cars).
    *   `--no-verbose`: Suppress detailed print output for each image when processing directories (shows progress bar instead).
*   **Example Commands:**
    *   Run AdaptiVision on a single image using MPS:
        ```bash
        python src/adaptivision.py --input_path samples/zidane.jpg --output_dir results/single_test --adaptive --context --device mps
        ```
    *   Run standard detection on a directory of images using CPU, saving to `results/standard_run`:
        ```bash
        python src/adaptivision.py --input_path datasets/coco/images/val2017/ --output_dir results/standard_run --device cpu --no-verbose
        ```

### 2. `scripts/save_coco_results.py`

*   **Purpose:** Runs object detection (either baseline YOLO or AdaptiVision) on all images in a specified dataset directory and saves the detection results in the standard COCO JSON format required for official evaluation tools (`pycocotools`). If running AdaptiVision, it also saves a separate JSON file containing the calculated scene complexity for each image.
*   **When to Use:** This is the first step in the pipeline for calculating standard evaluation metrics like mAP. Use this to generate the prediction file needed by `evaluate_coco.py`.
*   **Key Options:**
    *   `--dataset-path`: (Required) Path to the directory containing the dataset images (e.g., `datasets/coco/images/val2017/`).
    *   `--gt-annotations`: (Required) Path to the COCO ground truth annotation file (e.g., `datasets/coco/annotations/instances_val2017.json`). Needed for mapping filenames to image IDs.
    *   `--output-json`: (Required) Path where the output COCO-formatted prediction JSON file will be saved (e.g., `results/predictions/adaptivision_preds.json`).
    *   `--weights`: Path to the model weights file (default: `weights/model_n.pt`).
    *   `--method`: (Required) Choose the detection method: 'baseline' (standard YOLO) or 'adaptivision'.
    *   `--device`: Processing device ('auto', 'cpu', 'cuda', 'mps') (default: 'auto').
    *   `--baseline-conf`: Confidence threshold *only* for the initial low-confidence pass when `--method baseline` is used for mAP calculation (default: 0.001). Not used by `adaptivision`.
    *   `--adaptive`/`--no-adaptive`: Enable/disable adaptive confidence (default: enabled for `adaptivision` method).
    *   `--context`/`--no-context`: Enable/disable context reasoning (default: enabled for `adaptivision` method).
    *   `--postprocess-filter`/`--no-postprocess-filter`: Enable/disable geometric post-processing filter (default: enabled for `adaptivision` method).
*   **Output Files:**
    *   `<output-json>`: Contains detections in COCO format.
    *   `<output-json based name>_complexity.json`: (Only if `--method adaptivision`) Contains scene complexity scores per image ID.
*   **Example Commands:**
    *   Generate AdaptiVision predictions for COCO val set using MPS:
        ```bash
        python scripts/save_coco_results.py \
          --dataset-path datasets/coco/images/val2017/ \
          --gt-annotations datasets/coco/annotations/instances_val2017.json \
          --weights weights/model_n.pt \
          --output-json results/coco_eval/adaptivision_preds.json \
          --method adaptivision \
          --device mps
        ```
    *   Generate baseline YOLO predictions:
        ```bash
        python scripts/save_coco_results.py \
          --dataset-path datasets/coco/images/val2017/ \
          --gt-annotations datasets/coco/annotations/instances_val2017.json \
          --weights weights/model_n.pt \
          --output-json results/coco_eval/baseline_preds.json \
          --method baseline \
          --device mps
        ```
    *   Generate AdaptiVision predictions *without* context reasoning:
        ```bash
        python scripts/save_coco_results.py \
          --dataset-path datasets/coco/images/val2017/ \
          --gt-annotations datasets/coco/annotations/instances_val2017.json \
          --weights weights/model_n.pt \
          --output-json results/coco_eval/adaptivision_no_context_preds.json \
          --method adaptivision \
          --device mps \
          --no-context
        ```

### 3. `scripts/evaluate_coco.py`

*   **Purpose:** Takes a COCO-formatted ground truth annotation file and a COCO-formatted prediction file (generated by `save_coco_results.py`) and calculates standard COCO evaluation metrics using `pycocotools`. It prints overall AP/AR metrics, per-class AP, generates a Precision-Recall curve, and optionally performs evaluation broken down by scene complexity if a corresponding `_complexity.json` file is found.
*   **When to Use:** The second step in the mAP evaluation pipeline, after running `save_coco_results.py`. Use this to get quantitative performance metrics.
*   **Key Options:**
    *   `--annotation-file`: (Required) Path to the COCO ground truth JSON file.
    *   `--results-file`: (Required) Path to the prediction results JSON file generated by `save_coco_results.py`.
    *   `--iou-type`: Type of evaluation ('bbox', 'segm', 'keypoints') (default: 'bbox').
*   **Input Files:**
    *   Ground Truth JSON (`--annotation-file`).
    *   Prediction JSON (`--results-file`).
    *   (Optional) `<results-file based name>_complexity.json`: Used for complexity breakdown.
*   **Output:**
    *   Prints evaluation metrics to the console.
    *   Saves a Precision-Recall curve image (e.g., `..._pr_curve.png`) in the same directory as the results file.
*   **Example Command:**
    ```bash
    python scripts/evaluate_coco.py \
      --annotation-file datasets/coco/annotations/instances_val2017.json \
      --results-file results/coco_eval/adaptivision_preds.json
    ```

### 4. `scripts/run_experiments.py`

*   **Purpose:** Performs a comprehensive comparison between standard YOLO detection and AdaptiVision. It runs both methods on a dataset, collects detailed timing and detection statistics for each image, generates comparison visualizations, and saves aggregated results suitable for plotting and further analysis.
*   **When to Use:** Use this script when you want a detailed, head-to-head comparison of the performance (speed, detection counts, complexity effects) between the baseline and adaptive approaches across a dataset. It's more focused on comparative analysis than just getting a single mAP score.
*   **Key Options:**
    *   `--data`: (Required) Directory containing input images.
    *   `--output`: (Required) Base directory to save all experiment results and visualizations.
    *   `--weights`: Path to the model weights file (default: `weights/model_n.pt`).
    *   `--device`: Processing device ('auto', 'cpu', 'cuda', 'mps') (default: 'auto').
    *   `--reanalyze-only`: If present, skips running predictions and only re-runs the analysis/plotting steps based on an existing `detailed_results.json` file in the output directory.
*   **Output Structure (within `--output` dir):**
    *   `standard/`: Annotated images from standard detection.
    *   `adaptive/`: Annotated images from adaptive detection.
    *   `comparisons/`: Side-by-side comparison images.
    *   `visualizations/<image_stem>/`: Detailed visualizations per image (complexity, thresholds).
    *   `analytics/`: Plots and summary CSV files.
    *   `detailed_results.json`: Raw data for every image (timings, counts, scores, etc.).
    *   `summary_results.csv`: Aggregated results per image.
*   **Example Command:**
    ```bash
    python scripts/run_experiments.py \
      --data samples/coco/ \
      --output results/coco_experiment_comparison \
      --weights weights/model_n.pt \
      --device mps
    ```

### 5. Plotting Scripts (`generate_*.py`)

*   **Purpose:** These scripts (`generate_capped_time_plot.py`, `generate_overhead_plot.py`, etc.) are designed to consume the `summary_results.csv` file generated by `run_experiments.py` and create specific plots for analysis (e.g., processing time comparison, time overhead distribution).
*   **When to Use:** After running `run_experiments.py`, use these to generate publication-ready or analysis-focused plots based on the collected summary data.
*   **Configuration:** These scripts typically have configuration variables at the top (e.g., `RESULTS_CSV`, `OUTPUT_DIR`) that need to point to the correct output files from `run_experiments.py`.
*   **Example Command (assuming config is set correctly):**
    ```bash
    python scripts/generate_overhead_plot.py
    ```
    ```bash
    python scripts/generate_capped_time_plot.py
    ``` 