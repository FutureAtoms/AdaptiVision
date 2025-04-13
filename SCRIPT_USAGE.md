# AdaptiVision Script Usage and Testing Guide

This document details the purpose, usage, and testing status of the various Python scripts included in the AdaptiVision project.

**Note:** All tests were run using the Python interpreter from the project's virtual environment (`/Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python`) to ensure all dependencies were met.

## Quick Script Selection Guide

*   **For quick predictions and visual output on single images or directories:** Use `src/cli.py detect` or `src/cli.py batch`.
*   **For generating standard COCO evaluation metrics (like mAP):** Use the `scripts/save_coco_results.py` + `scripts/evaluate_coco.py` pipeline.
*   **For a detailed comparison between standard YOLO and AdaptiVision with ablation studies and analytics:** Use `scripts/run_experiments.py`.
*   **For generating specific plots from experiment results:** Use `scripts/generate_..._plot.py` scripts.
*   **For simple code examples:** See `examples/basic_detection.py` and `examples/batch_processing.py`.

## Detailed Script Descriptions

### 1. `src/cli.py` (Main Command-Line Interface)

*   **Purpose**: Provides a unified command-line interface (`adaptivision` if installed, or `python src/cli.py`) for various AdaptiVision functionalities.
*   **Dependencies**: `requirements.txt`, valid model weights (usually), valid input image/directory (depending on subcommand).
*   **Entry Point**: `python src/cli.py <command> [options]` or `adaptivision <command> [options]` if installed.

#### 1.1 `detect` Subcommand

*   **Purpose**: Perform detection on a single image.
*   **Key Options**: `--image` (required), `--output`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, `--disable-adaptive`, `--disable-context`.
*   **Output**: Single annotated output image specified by `--output`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py detect --image samples/bus.jpg --output results/test_cli_detect.jpg
    ```
*   **Example Command**:
    ```bash
    python src/cli.py detect --image path/to/image.jpg --output results/cli_detection.jpg
    ```
*   **Test Status**: **Passed**

#### 1.2 `compare` Subcommand

*   **Purpose**: Generate a side-by-side comparison image of standard vs. adaptive detection for a single input image.
*   **Key Options**: `--image` (required), `--output-dir`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`.
*   **Output**: Saves `comparison_<image_filename>.jpg` in the directory specified by `--output-dir`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py compare --image samples/bus.jpg --output-dir results/test_cli_compare
    ```
*   **Example Command**:
    ```bash
    python src/cli.py compare --image path/to/image.jpg --output-dir results/my_comparisons
    ```
*   **Test Status**: **Passed**

#### 1.3 `visualize` Subcommand

*   **Purpose**: Create detailed visualizations (complexity map, threshold map, metadata) related to the adaptive process for a single input image.
*   **Key Options**: `--image` (required), `--output-dir`, `--weights`, `--device`.
*   **Output**: Saves `complexity_<image_filename>.jpg`, `threshold_map_<image_filename>.jpg`, and `metadata_<image_stem>.json` in the directory specified by `--output-dir`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py visualize --image samples/bus.jpg --output-dir results/test_cli_visualize
    ```
*   **Example Command**:
    ```bash
    python src/cli.py visualize --image path/to/image.jpg --output-dir results/my_visualizations
    ```
*   **Test Status**: **Passed**

#### 1.4 `batch` Subcommand

*   **Purpose**: Process a directory of images in batch mode via the main CLI.
*   **Key Options**: `--input-dir` (required), `--output-dir`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, `--workers`, `--save-json`, `--disable-adaptive`, `--disable-context`.
*   **Output**: Annotated images and optional JSON files in the `--output-dir`.
*   **Test Commands Used**:
    ```bash
    # On samples/coco (after image replacement)
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py batch --input-dir samples/coco --output-dir results/test_cli_batch --save-json --workers 2

    # On samples/batch (after image replacement)
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py batch --input-dir samples/batch --output-dir results/test_batch --save-json --workers 2
    ```
*   **Example Command**:
    ```bash
    python src/cli.py batch --input-dir path/to/images --output-dir results/cli_batch_output --save-json
    ```
*   **Test Status**: **Passed** (after replacing corrupted sample images)

---

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
*   **Test Status**: Not explicitly tested in `script_usage_and_tests.md`, assumed functional as part of the evaluation pipeline.

---

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
*   **Test Status**: Not explicitly tested in `script_usage_and_tests.md`, assumed functional as part of the evaluation pipeline.

---

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
    *   `analytics/`: Plots and summary CSV/JSON files.
    *   `detailed_results.json`: Raw data for every image (timings, counts, scores, etc.).
    *   `summary_results.csv`: Aggregated results per image.
    *   `experiment_report.md`: Markdown report summarizing the overall experiment.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python scripts/run_experiments.py --data samples/coco --output results/test_run_experiment_mps_fixed --weights weights/model_n.pt --device mps
    ```
*   **Example Command**:
    ```bash
    python scripts/run_experiments.py \
      --data samples/coco/ \
      --output results/coco_experiment_comparison \
      --weights weights/model_n.pt \
      --device mps
    ```
*   **Test Status**: **Passed**
*   **Notes**: Primary script for full dataset experiments. Requires correct `sys.path` setup and robust image handling.

---

### 5. Plotting Scripts (`scripts/generate_*.py`)

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
*   **Test Status**: Not explicitly tested, assumed functional.

---

### 6. `examples/basic_detection.py`

*   **Purpose**: Demonstrates basic single-image object detection using the `AdaptiVision` class, including optional visualization and printing results to the console.
*   **Key Options**: `--image` (required), `--output`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, `--disable-adaptive`, `--disable-context`.
*   **Output**: Prints detection summary, saves annotated image, attempts to display image.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python examples/basic_detection.py --image samples/bus.jpg --output results/test_basic_detection.jpg
    ```
*   **Example Command**:
    ```bash
    python examples/basic_detection.py --image path/to/your/image.jpg --output results/my_detection.jpg
    ```
*   **Test Status**: **Passed**
*   **Notes**: Display logic (`cv2.waitKey(0)`) might cause interruptions in some environments.

---

### 7. `examples/batch_processing.py`

*   **Purpose**: Processes a directory of images in batch mode, optionally using parallel workers and saving detailed detection results as individual JSON files alongside output images.
*   **Key Options**: `--input-dir` (required), `--output-dir`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, `--workers`, `--image-types`, `--save-json`, `--disable-adaptive`, `--disable-context`.
*   **Output**: Annotated images, optional `.json` files, prints summary statistics.
*   **Test Command Used**:
    ```bash
    # Tested on samples/batch directory after image replacement
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python examples/batch_processing.py --input-dir samples/batch --output-dir results/test_batch --save-json --workers 2
    ```
*   **Example Command**:
    ```bash
    python examples/batch_processing.py --input-dir path/to/your/images --output-dir path/to/batch_results --save-json --workers 4
    ```
*   **Test Status**: **Passed** (after replacing corrupted sample images)
*   **Notes**: Parallel processing logic works well.

---

### 8. `src/compare_methods.py` (Internal Use / Direct Execution)

*   **Purpose**: Compares standard vs. adaptive detection for a single image. Primarily called by `src/cli.py compare`, but can be run directly.
*   **Direct Execution Options**: `--image` (required), `--output-dir`, `--weights`, `--conf`, `--iou`, `--device`.
*   **Output**: Saves `comparison_<image_filename>.jpg`.
*   **Test Status**: **Passed** (indirectly via `cli compare` test)
*   **Notes**: Requires correct `sys.path` if run directly.

---

### 9. `src/create_visualizations.py` (Internal Use / Direct Execution)

*   **Purpose**: Generates detailed visualizations for a single image. Primarily called by `src/cli.py visualize`, but can be run directly.
*   **Direct Execution Options**: `--image` (required), `--output-dir`, `--weights`, `--device`.
*   **Output**: Saves `complexity_*.jpg`, `threshold_map_*.jpg`, `metadata_*.json`.
*   **Test Status**: **Passed** (indirectly via `cli visualize` test)
*   **Notes**: Requires correct `sys.path` if run directly.

---

### 10. `src/adaptivision.py` (Deprecated Standalone Script?)

*   **Note:** The original `SCRIPT_USAGE.md` described a standalone `src/adaptivision.py` script for single/directory processing. This functionality seems to have been integrated into `src/cli.py` (`detect` and `batch` subcommands). The direct execution of `src/adaptivision.py` might be deprecated or intended only for library use. Use `src/cli.py` instead for command-line operations.

---

## Overall Test Summary

All tested scripts function correctly when provided with valid input data and run within the correct Python environment (`venv`). Issues related to environment setup, internal imports, visualization logic, and corrupted sample images were identified and fixed during the testing process documented in `script_usage_and_tests.md`. 