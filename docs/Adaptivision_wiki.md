# AdaptiVision System Wiki

## 1. Overview

AdaptiVision is an object detection system designed to enhance the robustness and accuracy of standard object detectors, specifically using YOLOv8n as its base model in this implementation. It achieves this by incorporating several layers of post-processing intelligence that adapt the detection process based on the characteristics of the input scene and contextual information. The primary goal is to improve the balance between detection recall (finding true objects) and precision (avoiding false positives) across diverse environments, ranging from simple scenes with few objects to complex, cluttered scenes.

The system operates by first running a base detector (YOLO) with a low initial confidence threshold to generate candidate detections. It then applies a series of refinement steps: scene complexity analysis, adaptive confidence thresholding, class-specific adjustments, context-aware reasoning, and final post-processing filters.

## 2. Core Innovations

AdaptiVision introduces several key mechanisms, primarily implemented in `src/adaptivision.py`:

### 2.1. Dynamic Scene Complexity Analysis

-   **Purpose:** To quantify how "complex" or "simple" an input scene is.
-   **Implementation (`_calculate_scene_complexity`):**
    -   Calculates a score between 0 and 1 based on three factors derived from initial, low-threshold detections:
        1.  **Number of Objects:** The count of detected objects. Log-scaled (`np.log2(num_objects + 1) / 5.0`) to handle large numbers more gracefully and normalized. Weighted by `complexity_factors['num_objects']` (default: 0.5).
        2.  **Object Size Variance:** The standard deviation of the bounding box areas, normalized by the mean area (`np.std(areas) / (np.mean(areas) + 1e-5)`). This measures the diversity in object sizes. Weighted by `complexity_factors['object_size_var']` (default: 0.25).
        3.  **Object Density:** The proportion of the image area covered by bounding boxes (`sum(areas) / image_area`). Weighted by `complexity_factors['object_density']` (default: 0.25).
-   **Output:** A single floating-point `scene_complexity` score. Higher values indicate more complex scenes (more objects, higher size variance, higher density).

### 2.2. Adaptive Confidence Thresholding

-   **Purpose:** To dynamically adjust the detection confidence threshold based on scene complexity, improving the precision/recall trade-off.
-   **Implementation (`_get_adaptive_threshold`):**
    -   Takes the `base_threshold` (e.g., 0.25) and the calculated `scene_complexity`.
    -   Applies an adjustment:
        -   **High Complexity (`> 0.7`):** Threshold is lowered significantly (up to `base - 0.12`). Aims to increase recall in cluttered scenes.
        -   **Low Complexity (`< 0.3`):** Threshold is raised (up to `base + 0.05`). Aims to increase precision in simple scenes.
        -   **Mid Complexity (`0.3` to `0.7`):** A non-linear interpolation between the min and max adjustments is applied.
    -   The final `adaptive_threshold` is clamped between 0.08 and 0.95.
-   **Effect:** Detections are primarily filtered based on this dynamic `adaptive_threshold` rather than a fixed one.

### 2.3. Class-Specific Adjustments

-   **Purpose:** To inject prior knowledge about specific object classes to fine-tune detection sensitivity.
-   **Implementation:**
    -   Uses a predefined dictionary `self.class_conf_adjustments`.
    -   Contains adjustments applied *additively* to the `adaptive_threshold` for specific classes *before* filtering.
    -   Examples:
        -   `'cell phone': -0.03` (requires lower score)
        -   `'car': +0.03` (requires higher score)
        -   `'stop sign': -0.03` (requires lower score - important object)
-   **Effect:** Modifies the effective threshold on a per-class basis, making the system more sensitive to certain (e.g., small, important) objects and less sensitive to others (e.g., large, common).

### 2.4. Context-Aware Reasoning (Object Relationships)

-   **Purpose:** To use the co-occurrence of objects in the scene to refine confidence scores, boosting likely true positives and penalizing potential false positives that lack context.
-   **Implementation (`_apply_context_reasoning`):**
    -   Uses a predefined dictionary `self.object_relationships` mapping an object class to a list of classes it commonly appears with (e.g., `'chair': ['dining table', 'person', ...]`).
    -   Calculates a `context_boost` (or penalty) for each detected object:
        1.  **Base Boost:** Small positive boost for important traffic objects (`stop sign`, `traffic light`, `bicycle`).
        2.  **Related Object Count:** Counts how many other detected objects in the scene belong to the current object's list of related classes.
        3.  **Isolation Penalty:** If `related_count` is 0 *and* the object is in the `rarely_alone_objects` list (e.g., 'tie', 'fork', 'handbag'), a negative boost (penalty) is applied (e.g., -0.05).
        4.  **Co-occurrence Boost:** If `related_count > 0`, a positive boost (up to +0.09) is calculated. The magnitude depends on the `related_count` relative to the total number of potential related objects and class-specific `context_requirements`.
    -   The `context_boost` is added to the object's original confidence score *before* it is compared against the class-adjusted adaptive threshold.
-   **Effect:** Increases the likelihood of keeping detections that fit the expected semantic context and decreases the likelihood of keeping those that appear out of context.

### 2.5. Refined Post-Processing

-   **Purpose:** A final filtering stage to remove common types of false positives based on geometric properties and class-specific rules.
-   **Implementation (`_post_process_detections`):**
    -   Applied *after* the adaptive thresholding and context reasoning.
    -   Removes detections based on:
        -   **Class Constraints:** Checks aspect ratio, relative image area, and minimum score thresholds for specific "problematic" classes (e.g., 'tie', 'cell phone', 'handbag', 'bicycle').
        -   **Small Object Filter:** Removes very small objects (`area < 0.001 * image_area`) if their score is low (threshold depends on whether it's an "important" class).
        -   **Boundary Proximity:** Penalizes objects touching image boundaries (score threshold depends on whether it's an "important" class).
-   **Effect:** Cleans up the final detection set by removing detections with unlikely shapes, sizes, or positions for their class.

## 3. Implementation Details

-   **Base Model:** Uses `ultralytics YOLO` (specifically, weights from `weights/model_n.pt` suggest YOLOv8n by default). Loaded in the `AdaptiVision.__init__` method.
-   **Core Logic:** Primarily contained within the `AdaptiVision` class in `src/adaptivision.py`.
-   **Prediction Pipeline (`predict` method):**
    1.  Loads image (handles path or NumPy array, includes PIL fallback).
    2.  Runs initial YOLO inference with a very low confidence threshold (`initial_conf`).
    3.  Calculates scene complexity.
    4.  Calculates adaptive threshold.
    5.  Iterates through initial detections:
        -   Applies class-specific threshold adjustment.
        -   Calculates context boost and adds it to the score.
        -   Filters based on the class-adjusted adaptive threshold and boosted score.
    6.  Applies `_post_process_detections` for final filtering.
    7.  Returns results (bounding boxes, scores, labels, class names, metadata).
-   **Helper Functions (`src/utils.py`):** Contains standard utilities for device selection, image loading/preprocessing (`load_image`, `resize_and_pad`), bounding box coordinate adjustments (`adjust_boxes_to_original`), and visualization (`draw_detections`). The visualization function includes logic to visually indicate the effect of the adaptive threshold.
-   **Dependencies:** `ultralytics`, `numpy`, `opencv-python`, `torch`, `Pillow` (optional fallback for image loading). Listed in `requirements.txt`.

## 4. Evaluation Strategy

-   **Qualitative Comparison (`src/compare_methods.py`):**
    -   Provides a script to run both the baseline (standard fixed threshold) and the full AdaptiVision system on an input image.
    -   Generates a side-by-side output image (`results/comparison/comparison_*.jpg`) visualizing the detections from both methods.
    -   Displays the scene complexity and adaptive threshold used by AdaptiVision.
    -   Useful for visually understanding the system's behavior in different scenarios.
-   **Quantitative Evaluation (`results/map_analysis/`):**
    -   Contains `baseline_val2017_predictions.json` and `adaptivision_val2017_predictions.json`.
    -   These files indicate that the system was evaluated against its baseline on the standard COCO val2017 benchmark dataset.
    -   The results would typically be processed using tools like `pycocotools` to calculate standard object detection metrics (mAP, AP50, AP75, AR, etc.), although the specific scores are not directly present in the repository structure explored. The research paper (`research_paper/adaptivision_paper.pdf`) likely contains these detailed results.

## 5. Comparison and Context (Relation to State-of-the-Art)

-   **Adaptive Thresholding:** This concept exists in research, but AdaptiVision's specific approach using scene complexity metrics (count, size variance, density) calculated from initial detections appears distinct from methods based on distance or tracker-internal score distributions.
-   **Contextual Reasoning:** While using context is common in SOTA models (often via attention or GNNs integrated into the architecture), AdaptiVision uses a simpler, explicit, rule-based post-processing approach based on object co-occurrence lists. This is more interpretable but potentially less powerful/generalizable than end-to-end learned context.
-   **Overall Approach:** AdaptiVision focuses on adding intelligence via *post-processing* modules on top of a standard detector, contrasting with methods that modify the core detector architecture.

## 6. Strengths and Potential Weaknesses

### Strengths

-   **Improved Robustness:** Aims to provide better performance across a wider range of scene complexities compared to a fixed-threshold detector.
-   **Interpretability:** The rules governing complexity, threshold adaptation, and context are relatively explicit and easier to understand than complex neural network mechanisms.
-   **Modularity:** Built as layers on top of a standard YOLO model, potentially allowing the techniques to be applied to other base detectors.
-   **Targeted Problem Solving:** Directly addresses known issues like precision/recall trade-offs in varying scenes and contextual ambiguity.
-   **Quantitatively Validated:** Evidence of evaluation on the standard COCO benchmark exists.

### Potential Weaknesses / Trade-offs

-   **Reliance on Heuristics/Tuning:** The effectiveness depends heavily on the quality and tuning of the hand-crafted rules, weights, thresholds, and dictionaries (complexity factors, class adjustments, context relationships, post-processing parameters). These might require careful adjustment for different datasets or domains.
-   **Context Scalability:** The explicit `object_relationships` dictionary could become difficult to create and maintain for datasets with a very large number of classes or intricate relationships.
-   **Performance vs. Integrated SOTA:** While improving over its baseline, it might not achieve the peak performance of state-of-the-art detectors that incorporate adaptive or contextual mechanisms directly into their neural network architecture through end-to-end training.
-   **Computational Overhead:** Adds extra computational steps after the initial YOLO inference, although these are likely minor compared to the main detection network.

## 7. Usage (Command Line Interface)

The system can be run using `src/adaptivision.py` or potentially through `src/cli.py`. Key arguments (based on `adaptivision.py` main section and `script_usage_and_tests.md`):

-   `--image`: Path to the input image (required).
-   `--weights`: Path to the YOLO model weights (default: `weights/model_n.pt`).
-   `--output`: Path to save the output visualization image (default: `output/detected.jpg`).
-   `--conf`: Base confidence threshold (default: 0.25). Used directly if adaptive/context are off, or as the base for adaptive adjustments.
-   `--iou`: IoU threshold for Non-Maximum Suppression (default: 0.45).
-   `--adaptive`: Flag to **enable** adaptive confidence thresholding.
-   `--context`: Flag to **enable** context-aware reasoning.
-   `--device`: Device to run on ('auto', 'cpu', 'cuda', 'mps', default: 'auto').
-   `--classes`: List of specific class IDs to filter for (optional).

Example (running full AdaptiVision):

```bash
python src/adaptivision.py --image path/to/your/image.jpg --output results/output.jpg --adaptive --context
```

Example (running baseline):

```bash
python src/adaptivision.py --image path/to/your/image.jpg --output results/baseline_output.jpg --conf 0.25
```

The comparison script (`src/compare_methods.py`) uses similar arguments to generate the side-by-side visualization.

## 8. Conclusion

AdaptiVision enhances a standard YOLOv8n detector by introducing dynamic, context-sensitive post-processing steps. By analyzing scene complexity, adapting confidence thresholds, leveraging class-specific knowledge, and incorporating object co-occurrence context, it aims for more robust and reliable object detection across diverse visual environments. Its modular and interpretable design offers advantages, though its performance relative to end-to-end SOTA models depends on the quality of its heuristics and tuning. The presence of benchmark evaluation files confirms its quantitative assessment against its baseline.

## 9. Repository Structure

```
AdaptiVision/
├── .git/               # Git version control files
├── .gitignore          # Files/directories ignored by Git
├── adaptivision.egg-info/ # Python package build info
├── coco128_config.yaml # Configuration file (likely for dataset/training)
├── datasets/           # Likely placeholder or location for datasets
├── docs/
│   ├── Adaptivision_wiki.md # This wiki file
│   └── repo_analysis_overview.md # Initial brief analysis
├── examples/           # Example usage scripts or notebooks
├── LICENSE             # Project license file (e.g., MIT)
├── README.md           # Main project description, setup, usage
├── requirements.txt    # Python dependencies
├── research_paper/
│   ├── adaptivision_paper.pdf # PDF of the research paper detailing the method
│   └── ...               # LaTeX source files for the paper
├── results/
│   ├── comparison/       # Output directory for compare_methods.py visualizations
│   ├── map_analysis/     # Directory containing quantitative evaluation results (JSON files)
│   └── ...               # Other potential result directories (e.g., video, live)
├── samples/            # Sample input images/videos for testing
├── scripts/            # Utility or helper scripts
├── script_usage_and_tests.md # Notes on script usage/testing
├── setup.py            # Python package setup script
├── src/
│   ├── adaptivision.py   # Core AdaptiVision class and main execution logic
│   ├── cli.py            # Command-line interface handler (likely uses adaptivision.py)
│   ├── compare_methods.py # Script for side-by-side comparison visualization
│   ├── create_visualizations.py # Script for generating visualizations (potentially more advanced)
│   ├── utils.py          # Helper functions (image loading, drawing, etc.)
│   └── ...               # Other potential source files (e.g., video processing)
├── venv/               # Virtual environment (if used, typically gitignored)
└── weights/
    └── model_n.pt        # Default YOLOv8n model weights
```

## 10. Dependencies and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url> # Replace with actual URL
    cd AdaptiVision
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    The primary dependencies are listed in `requirements.txt`. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    Key dependencies include:
    -   `ultralytics`: For the base YOLOv8 model.
    -   `numpy`: For numerical operations.
    -   `opencv-python`: For image loading, processing, and drawing.
    -   `torch`: The deep learning framework used by YOLO.
    -   `Pillow`: Used as a fallback for image loading if OpenCV fails.

4.  **Download Model Weights (if needed):**
    The default weights (`weights/model_n.pt`) are expected to be present. If not, they might need to be downloaded separately based on instructions likely found in the main `README.md` or from the Ultralytics documentation.

5.  **Run the Package Setup (Optional, for development/distribution):**
    ```bash
    pip install -e .
    ```
    This installs the package in editable mode.

## 11. Key Tunable Parameters (within `src/adaptivision.py`)

Beyond the command-line arguments, several internal parameters within the `AdaptiVision` class constructor (`__init__`) define the system's behavior and can be tuned for different performance characteristics or datasets:

-   **`self.complexity_factors` (dict):**
    -   Weights controlling the contribution of object count, size variance, and density to the `scene_complexity` score.
    -   Defaults: `{'num_objects': 0.5, 'object_size_var': 0.25, 'object_density': 0.25}`.
    -   *Tuning:* Adjusting these weights changes how scene complexity is perceived. Increasing `num_objects` weight makes the system more sensitive to object count, etc.

-   **`self.class_conf_adjustments` (dict):**
    -   Per-class adjustments added to the adaptive threshold.
    -   Positive values increase the required score (higher precision), negative values decrease it (higher recall).
    -   *Tuning:* Requires domain knowledge. Add/modify entries for classes specific to a new dataset or adjust values based on observed class-specific performance (e.g., if 'trucks' are often missed, consider a negative adjustment).

-   **`self.object_relationships` (dict):**
    -   Defines expected co-occurrences for context-aware reasoning.
    -   Maps a class name to a list of other class names typically found nearby.
    -   *Tuning:* Crucial for the context module. Needs to be adapted based on the environment and dataset (e.g., indoor vs. outdoor scenes will have different relationships).

-   **`self.rarely_alone_objects` (dict, within `_apply_context_reasoning`):**
    -   Defines objects that receive a confidence penalty if detected without context.
    -   Maps class name to the penalty value.
    -   *Tuning:* Add/remove classes or adjust penalties based on how often objects truly appear in isolation in the target domain.

-   **`self.context_requirements` (dict, within `_apply_context_reasoning`):**
    -   Specifies the proportion of related objects needed for the maximum context boost for certain classes.
    -   *Tuning:* Adjusting these values changes how strong the contextual evidence needs to be to grant a significant confidence boost.

-   **Adaptive Threshold Parameters (within `_get_adaptive_threshold`):**
    -   `min_adjust`: Maximum threshold reduction (default: -0.12).
    -   `max_adjust`: Maximum threshold increase (default: +0.05).
    -   Complexity ranges (0.3, 0.7) and the non-linear mapping function.
    -   Threshold clamping bounds (0.08, 0.95).
    -   *Tuning:* Modifying these directly changes the sensitivity and range of the adaptive threshold mechanism.

-   **Post-processing Constraints (`problematic_classes` dict and logic within `_post_process_detections`):**
    -   Aspect ratio, area, and score limits for specific classes.
    -   Thresholds for small objects and boundary-touching objects.
    -   *Tuning:* Adjust these constraints based on observed false positive patterns (e.g., if many thin false positive 'poles' are detected, adjust aspect ratio constraints).

The system was quantitatively evaluated using the COCO val2017 dataset (mentioned in Section 4).
The documentation doesn't explicitly state the specific input image size(s) used by the AdaptiVision pipeline during this evaluation or for the base YOLO model.
The COCO dataset itself contains images of varying resolutions. They are not standardized to a single size in the original dataset.
Object detection models typically resize images to a fixed square size (e.g., 640x640, 1280x1280) for processing. The wiki confirms this practice by mentioning a resize_and_pad helper function in src/utils.py (Section 3).
Therefore, while the evaluation used COCO val2017, the images were likely resized to a fixed dimension required by the YOLOv8n model before being processed by the AdaptiVision system. The exact size isn't specified in the wiki, but 640x640 is a common default for YOLOv8 models.


*Note:* Tuning these internal parameters likely requires re-running evaluations (qualitative and quantitative) to assess the impact on performance. 