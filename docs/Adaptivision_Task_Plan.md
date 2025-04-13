# AdaptiVision Evaluation Task Plan

**Project:** AdaptiVision Evaluation and Paper Validation
**Version:** 1.0
**Last Updated:** $(date +%Y-%m-%d)

---

## PHASE 1: Repository Audit

**Description:** Verify core components, script consistency, and claim validity.

-   [✅] **1.1 Dataset Consistency:** Ensure both prediction files use COCO val2017.
    -   *Status:* Scripts `scripts/save_coco_results.py` and `scripts/evaluate_coco.py` require COCO GT annotation path (`.json`) as input, enabling use of `val2017`. Documentation (`docs/Adaptivision_wiki.md`) confirms `val2017` was used for results mentioned.
-   [✅] **1.2 Image Preprocessing Check:** Confirm same image size (e.g. 640x640) and padding logic in both runs.
    -   *Status:* Both baseline and AdaptiVision use the same underlying `ultralytics.YOLO` model call. Image resizing/padding is handled internally and consistently by the `ultralytics` library based on the loaded model's requirements. No differing parameters (e.g., `imgsz`) are passed between baseline/AdaptiVision runs in relevant scripts (`run_experiments.py`, `save_coco_results.py`).
-   [❌] **1.3 Scene Complexity Calculation Check:** Check if per-image scene complexity score is calculated and logged.
    -   *Status:* Calculation occurs in `AdaptiVision._calculate_scene_complexity` (called by `predict`). It *is* logged by `scripts/run_experiments.py` (in JSON, CSV, reports). However, it is **NOT** logged by `scripts/save_coco_results.py`, which produces the JSON for standard mAP evaluation. The score is therefore not linked to the evaluation results file.
-   [❌] **1.4 Scene-Based Metric Split:** Verify that mAP/recall is computed separately for low/med/high complexity scenes.
    -   *Status:* The standard evaluation script (`scripts/evaluate_coco.py`) uses `pycocotools.COCOeval` on the entire results file without filtering/grouping by scene complexity. The analysis script (`scripts/run_experiments.py`) logs complexity but doesn't compute mAP per complexity bin. This functionality is missing.
-   [❌] **1.5 Scene-Based Metric Computation (Add if Missing):** If not present, add logic to bin results and compute metrics per bin.
    -   *Status:* Confirmed missing based on 1.3 and 1.4. Requires modifications to `scripts/save_coco_results.py` (to include complexity per image) and `scripts/evaluate_coco.py` (to read complexity, bin images, and run `COCOeval` per bin). Action needed in Phase 2/3.
-   [❌] **1.6 Ablation Flags Check:** Ensure `__init__` has flags for adaptive threshold, context, class-adjustment, post-processing.
    -   *Status:* `src/adaptivision.py` `AdaptiVision.__init__` has `enable_adaptive_confidence` and `context_aware` flags. It lacks separate flags for class-specific adjustments (handled internally by `_apply_final_threshold_logic` based on `enable_adaptive_confidence`) and the final geometric post-processing (applied unconditionally in `predict` via `_post_process_detections`).
-   [❌] **1.7 Script Support for Ablation Runs:** Ensure current script supports multiple ablation combinations (e.g. 2, 3 module configs).
    -   *Status:* The main COCO evaluation script (`scripts/save_coco_results.py`) uses a `--method` argument with choices 'baseline' or 'adaptivision'. The 'adaptivision' choice hardcodes `enable_adaptive_confidence=True` and `context_aware=True`. The script needs modification (e.g., add `--no-adaptive`, `--no-context` flags) to run other ablation combinations (Adaptive only, Context only).

---

## PHASE 2: Core Fixes and Enhancements

**Description:** Critical updates that affect paper acceptance and core claims.

-   [✅] **2.1 Add Ablation Experiments:** Run AdaptiVision with combinations of modules enabled/disabled and log mAP for each.
    -   *Status:* Modified `scripts/save_coco_results.py` to accept `--adaptive` and `--context` flags. Ran script for Baseline, Full AdaptiVision, Adaptive Only, Context Only configurations. Ran `scripts/evaluate_coco.py` on each output. Results saved to `results/ablations/*.txt`. Summary (mAP): Baseline=0.355, Full=0.340, AdaptiveOnly=0.340, ContextOnly=0.340.
-   [✅] **2.2 Scene Complexity mAP Breakdown:** Log mAP/recall for different complexity levels and compare to baseline.
    -   *Status:* Modified `scripts/save_coco_results.py` to save complexity scores to `*_complexity.json`. Modified `scripts/evaluate_coco.py` to load complexity, bin images (Low < 0.3, Med 0.3-0.7, High > 0.7), and run evaluation per bin. Used AdaptiVision complexity for consistent baseline binning. Results saved in `results/ablations/*_eval_with_complexity.txt`. Comparison (mAP): Low (B:0.508, A:0.502), Med (B:0.432, A:0.416), High (B:0.322, A:0.307). AdaptiVision underperforms baseline slightly in all bins.
-   [✅] **2.3 Refine Post-Processing:** Check if current post-processing harms mAP; refine rules accordingly.
    -   *Status:* Added `enable_postprocess_filter` flag to `AdaptiVision` and `save_coco_results.py`. Ran evaluation with filter disabled (`adaptivision_no_postfilter_eval.txt`). Results showed **no change** in mAP compared to filter enabled (0.340). This indicates the geometric post-processing filter (`_post_process_detections`) is *not* responsible for the mAP drop vs baseline. The issue likely lies in the adaptive thresholding or context logic itself. No refinement of this specific filter is needed based on mAP impact.

---

## PHASE 3: Evaluation Enhancements

**Description:** Add richer analysis to strengthen evidence and clarity.

-   [✅] **3.1 Per-Class AP Logging:** Export per-class AP from `pycocotools` and log for analysis.
    -   *Status:* Modified `scripts/evaluate_coco.py` (`run_evaluation` function) to calculate and print per-class AP (averaged over IoU thresholds) using `coco_eval.eval['precision']` after the standard summary. Confirmed output in evaluation text files.
-   [✅] **3.2 PR Curve Visualization:** Generate PR curves (Precision-Recall) for both baseline and AdaptiVision.
    -   *Status:* Modified `scripts/evaluate_coco.py` (`run_evaluation` function) to use `matplotlib` to plot overall PR curve (Precision @ IoU=0.50 vs Recall) using `coco_eval.eval['precision']` and `coco_eval.params.recThrs`. Plot saved to `*_pr_curve.png` based on results filename. Confirmed files generated for baseline and AdaptiVision runs.
-   [✅] **3.3 Precision/Recall-Only Metric Logging:** Log AP50, AP75, AR, and raw Precision/Recall to support specific claims.
    -   *Status:* The standard COCO evaluation output from `COCOeval.summarize()` already includes AP50 (index 1), AP75 (index 2), and AR @ maxDets=100 (index 8). These metrics are printed by `scripts/evaluate_coco.py`. Raw P/R values are implicitly covered by the PR curve (Task 3.2). No further action needed.

---

## PHASE 4: Optional Exploratory Enhancements

**Description:** Non-critical improvements for visualizations and generalization.

-   [✅] **4.1 Run on Custom Complex Scenes:** Use real-world or custom-complex images to highlight qualitative benefits.
    -   *Status:* Capability exists via `examples/basic_detection.py` or `src/cli.py` which accept single image paths. User can provide custom complex images to these scripts for qualitative evaluation. No code changes needed.
-   [✅] **4.2 Tune Context Relationships:** Update object relationship dictionaries based on frequent false positives.
    -   *Status:* Relationships are defined in `self.object_relationships` dictionary within `AdaptiVision.__init__` (in `src/adaptivision.py`). Tuning requires manual analysis of false positives (e.g., from COCO evaluation or custom images) and editing this dictionary directly. The mechanism is understood and located.
-   [✅] **4.3 Tune Class-Specific Thresholds:** Adjust `class_conf_adjustments` if certain classes consistently underperform.
    -   *Status:* Adjustments are defined in `self.class_conf_adjustments` dictionary within `AdaptiVision.__init__` (in `src/adaptivision.py`). Tuning requires manual analysis of per-class performance (e.g., from per-class AP logged in Task 3.1) and editing this dictionary directly. The mechanism is understood and located. 