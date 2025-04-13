# Script Usage and Test Summary

This document details the usage and testing status of the main executable scripts in the AdaptiVision repository.

**Note:** All tests were run using the Python interpreter from the project's virtual environment (`/Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python`) to ensure all dependencies were met.

## 1. `scripts/run_experiments.py`

*   **Purpose**: Runs a comprehensive comparative experiment between standard and adaptive detection methods on all images in a specified data directory. It saves individual detection results, comparison images, detailed visualizations, and generates summary analytics (plots, CSV, markdown report).
*   **Dependencies**: 
    *   Python packages listed in `requirements.txt`.
    *   Valid model weights file (e.g., `weights/model_n.pt`).
    *   Input directory containing valid image files.
*   **Input Requirements**: 
    *   `--data`: Path to the input image directory (required).
    *   `--output`: Path to the output directory for results (required).
    *   `--weights`: Path to model weights file (default: `weights/model_n.pt`).
    *   `--device`: Compute device ('auto', 'cpu', 'cuda', 'mps') (default: 'auto').
*   **Output Structure** (within the specified `--output` directory):
    *   `standard/`: Directory with standard detection output images.
    *   `adaptive/`: Directory with adaptive detection output images.
    *   `comparisons/`: Directory with side-by-side comparison images.
    *   `visualizations/<image_stem>/`: Subdirectories for each image containing `complexity_*.jpg`, `threshold_map_*.jpg`, `metadata_*.json`.
    *   `analytics/`: Directory with summary plots (`.png`) and `summary_statistics.json`.
    *   `detailed_results.json`: JSON file with raw data for each processed image.
    *   `summary_results.csv`: CSV file summarizing key metrics per image.
    *   `experiment_report.md`: Markdown report summarizing the overall experiment.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python scripts/run_experiments.py --data samples/coco --output results/test_run_experiment_mps_fixed --weights weights/model_n.pt --device mps
    ```
*   **Example Command**:
    ```bash
    python scripts/run_experiments.py --data path/to/your/images --output path/to/your/results --device mps
    ```
*   **Test Status**: **Passed**
*   **Notes**: This script is the primary way to run full dataset experiments. Requires correct `sys.path` setup for internal imports, which was fixed during testing. Also requires fixes in comparison/visualization functions to handle image formats correctly.

## 2. `examples/basic_detection.py`

*   **Purpose**: Demonstrates basic single-image object detection using the `AdaptiVision` class, including optional visualization and printing results to the console.
*   **Dependencies**: 
    *   `requirements.txt`.
    *   Valid model weights file.
    *   Valid input image file.
*   **Input Requirements**: 
    *   `--image`: Path to the input image (required).
    *   `--output`: Path to save the output visualization image (default: `results/detection.jpg`).
    *   `--weights`: Path to model weights (default: `weights/model_n.pt`).
    *   `--conf-thres`, `--iou-thres`: Detection parameters.
    *   `--device`: Compute device.
    *   `--disable-adaptive`, `--disable-context`: Flags to turn off features.
*   **Output Structure**: 
    *   Prints detection summary to console.
    *   Saves a single annotated image to the path specified by `--output`.
    *   Attempts to display the image in a window (may hang or fail in headless environments).
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python examples/basic_detection.py --image samples/bus.jpg --output results/test_basic_detection.jpg
    ```
*   **Example Command**:
    ```bash
    python examples/basic_detection.py --image path/to/your/image.jpg --output results/my_detection.jpg
    ```
*   **Test Status**: **Passed**
*   **Notes**: Required code fixes during testing to handle dictionary keys and visualization return values correctly. Display logic using `cv2.waitKey(0)` might cause interruptions.

## 3. `examples/batch_processing.py`

*   **Purpose**: Processes a directory of images in batch mode, optionally using parallel workers and saving detailed detection results as individual JSON files alongside output images.
*   **Dependencies**: 
    *   `requirements.txt`.
    *   Valid model weights file.
    *   Input directory containing valid image files.
*   **Input Requirements**: 
    *   `--input-dir`: Path to the input image directory (required).
    *   `--output-dir`: Path to the output directory (default: `results/batch`).
    *   `--weights`, `--conf-thres`, `--iou-thres`, `--device`: Detection parameters.
    *   `--workers`: Number of parallel workers (default: 1).
    *   `--image-types`: Comma-separated image extensions (default: `jpg,jpeg,png`).
    *   `--save-json`: Flag to save detailed JSON output per image.
    *   `--disable-adaptive`, `--disable-context`: Flags to turn off features.
*   **Output Structure** (within the specified `--output-dir`):
    *   Annotated output image for each successfully processed input image.
    *   Optional `.json` file for each image containing detailed detection results if `--save-json` is used.
    *   Prints summary statistics to the console.
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
*   **Notes**: The script structure and parallel processing logic work correctly. Robust image loading (added during testing) is important.

## 4. `src/cli.py` (Main CLI Tool)

*   **Purpose**: Provides a unified command-line interface (`adaptivision` if installed, or `python src/cli.py`) for various AdaptiVision functionalities.
*   **Dependencies**: 
    *   `requirements.txt`.
    *   Valid model weights file (usually needed).
    *   Valid input image or directory (depending on subcommand).
*   **Entry Point**: Can be run as `python src/cli.py <command> [options]` or potentially as `adaptivision <command> [options]` if installed via `setup.py`.

### 4.1 `detect` Subcommand

*   **Purpose**: Perform detection on a single image.
*   **Input Requirements**: `--image` (required), `--output`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, flags.
*   **Output Structure**: Single annotated output image specified by `--output`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py detect --image samples/bus.jpg --output results/test_cli_detect.jpg
    ```
*   **Example Command**:
    ```bash
    python src/cli.py detect --image path/to/image.jpg --output results/cli_detection.jpg
    ```
*   **Test Status**: **Passed**

### 4.2 `compare` Subcommand

*   **Purpose**: Generate a side-by-side comparison image of standard vs. adaptive detection for a single input image.
*   **Input Requirements**: `--image` (required), `--output-dir`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`.
*   **Output Structure**: Saves `comparison_<image_filename>.jpg` in the directory specified by `--output-dir`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py compare --image samples/bus.jpg --output-dir results/test_cli_compare
    ```
*   **Example Command**:
    ```bash
    python src/cli.py compare --image path/to/image.jpg --output-dir results/my_comparisons
    ```
*   **Test Status**: **Passed**

### 4.3 `visualize` Subcommand

*   **Purpose**: Create detailed visualizations (complexity map, threshold map, metadata) related to the adaptive process for a single input image.
*   **Input Requirements**: `--image` (required), `--output-dir`, `--weights`, `--device`.
*   **Output Structure**: Saves `complexity_<image_filename>.jpg`, `threshold_map_<image_filename>.jpg`, and `metadata_<image_stem>.json` in the directory specified by `--output-dir`.
*   **Test Command Used**:
    ```bash
    /Users/abhilashchadhar/uncloud/Learn_apativision/AdaptiVision/venv/bin/python src/cli.py visualize --image samples/bus.jpg --output-dir results/test_cli_visualize
    ```
*   **Example Command**:
    ```bash
    python src/cli.py visualize --image path/to/image.jpg --output-dir results/my_visualizations
    ```
*   **Test Status**: **Passed**

### 4.4 `batch` Subcommand

*   **Purpose**: Process a directory of images in batch mode via the main CLI, essentially acting as a wrapper around the logic in `examples/batch_processing.py`.
*   **Input Requirements**: `--input-dir` (required), `--output-dir`, `--weights`, `--conf-thres`, `--iou-thres`, `--device`, `--workers`, `--save-json`, flags.
*   **Output Structure**: Same as `examples/batch_processing.py`: Annotated images and optional JSON files in the `--output-dir`.
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

## 5. `src/compare_methods.py` (Direct Execution)

*   **Purpose**: Compares standard vs. adaptive detection for a single image and saves a comparison image. Primarily intended to be called by `src/cli.py compare`, but its `main()` function allows direct execution.
*   **Dependencies**: `requirements.txt`, model weights, input image.
*   **Input Requirements (if run directly)**: `--image` (required), `--output-dir`, `--weights`, `--conf`, `--iou`, `--device`.
*   **Output Structure**: Saves `comparison_<image_filename>.jpg` in the `--output-dir`.
*   **Test Command Used**: Not tested directly. Tested via `src/cli.py compare`.
*   **Example Command (Direct)**:
    ```bash
    python src/compare_methods.py --image path/to/image.jpg --output-dir results/direct_compare --device mps
    ```
*   **Test Status**: **Passed** (indirectly via CLI test)
*   **Notes**: Requires correct `sys.path` setup for internal imports if run directly, which was fixed during testing.

## 6. `src/create_visualizations.py` (Direct Execution)

*   **Purpose**: Generates detailed visualizations (complexity map, threshold map, metadata) for a single image. Primarily intended to be called by `src/cli.py visualize`, but its `main()` function allows direct execution.
*   **Dependencies**: `requirements.txt`, model weights, input image.
*   **Input Requirements (if run directly)**: `--image` (required), `--output-dir`, `--weights`, `--device`.
*   **Output Structure**: Saves `complexity_<image_filename>.jpg`, `threshold_map_<image_filename>.jpg`, and `metadata_<image_stem>.json` in the `--output-dir`.
*   **Test Command Used**: Not tested directly. Tested via `src/cli.py visualize`.
*   **Example Command (Direct)**:
    ```bash
    python src/create_visualizations.py --image path/to/image.jpg --output-dir results/direct_visualize --device mps
    ```
*   **Test Status**: **Passed** (indirectly via CLI test)
*   **Notes**: Requires correct `sys.path` setup for internal imports if run directly, which was fixed during testing.

## Overall Summary

All tested scripts function correctly when provided with valid input data (especially images) and run within the correct Python environment (`venv`). Issues related to environment setup, internal imports, visualization logic, and corrupted sample images were identified and fixed during the testing process. 