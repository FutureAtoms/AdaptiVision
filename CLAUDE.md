# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AdaptiVision is an adaptive context-aware object detection system that dynamically adjusts confidence thresholds based on scene complexity and context awareness. It's built on top of YOLOv8 (via Ultralytics) and achieves up to 8.9× faster processing with improved detection quality.

**Core Innovation**: The system analyzes scene complexity and adjusts detection thresholds adaptively, with context-aware reasoning about object relationships to improve detection accuracy—especially for small objects like books and cell phones.

## Development Setup

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in editable mode with dependencies
pip install -e .

# CRITICAL: Install specific Ultralytics version (tested and verified)
pip install ultralytics==8.3.107

# Download model weights
mkdir -p weights
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o weights/model_n.pt
```

### Device Configuration
The system supports multiple compute devices:
- `auto`: Automatically selects CUDA > MPS > CPU
- `cuda`: NVIDIA GPUs
- `mps`: Apple Silicon (M1/M2/M3)
- `cpu`: CPU fallback

## Architecture

### Core Components

**`src/adaptivision.py`**: Main `AdaptiVision` class implementing:
- Scene complexity analysis based on object count, size variance, and density
- Adaptive threshold calculation (dynamic per scene)
- Context-aware reasoning using object relationship knowledge base
- Class-specific confidence adjustments (small objects get lower thresholds)
- Post-processing geometric validation filter

**`src/cli.py`**: Unified CLI with subcommands:
- `detect`: Single image detection
- `compare`: Side-by-side standard vs. adaptive comparison
- `visualize`: Generate complexity/threshold maps
- `batch`: Batch process directories with parallel workers

**`src/utils.py`**: Shared utility functions

**`src/compare_methods.py`**: Standard vs. adaptive comparison logic

**`src/create_visualizations.py`**: Visualization generation for complexity analysis

### Key Design Patterns

**Adaptive Confidence Logic**:
1. Run initial detection at very low threshold (0.05)
2. Calculate scene complexity from initial detections
3. Compute adaptive threshold based on complexity
4. Apply class-specific adjustments (small objects: -0.03 to -0.05, large objects: +0.03 to +0.05)
5. Apply context reasoning boosts (objects near related objects get +0.02 boost)
6. Filter with class-specific thresholds
7. Apply geometric post-processing to reduce false positives

**Object Relationships**: The system maintains a knowledge base in `self.object_relationships` defining which objects commonly appear together (e.g., 'person' with 'chair', 'cup', 'laptop').

## Common Development Tasks

### Running Detection

**Single Image**:
```bash
python src/cli.py detect \
  --image samples/bus.jpg \
  --output results/detection.jpg \
  --weights weights/model_n.pt \
  --device mps
```

**Batch Processing**:
```bash
python src/cli.py batch \
  --input-dir samples/coco/ \
  --output-dir results/batch/ \
  --weights weights/model_n.pt \
  --workers 2 \
  --save-json
```

**Standard YOLO (Disable Adaptive)**:
```bash
python src/cli.py detect \
  --image samples/bus.jpg \
  --disable-adaptive \
  --disable-context
```

### Running Experiments

**Full Experimental Comparison** (recommended for evaluation):
```bash
python scripts/run_experiments.py \
  --data datasets/coco128/images/train2017/ \
  --output results/experiment_name/ \
  --weights weights/model_n.pt \
  --device mps
```

This generates:
- `standard/` and `adaptive/` annotated images
- `comparisons/` side-by-side images
- `visualizations/` complexity/threshold maps per image
- `analytics/` plots and CSV summaries
- `detailed_results.json` raw data
- `summary_results.csv` aggregated results
- `experiment_report.md` summary report

### COCO Evaluation (mAP Metrics)

**Generate Predictions**:
```bash
# AdaptiVision predictions
python scripts/save_coco_results.py \
  --dataset-path datasets/coco/images/val2017/ \
  --gt-annotations datasets/coco/annotations/instances_val2017.json \
  --weights weights/model_n.pt \
  --output-json results/adaptivision_preds.json \
  --method adaptivision \
  --device mps

# Baseline predictions
python scripts/save_coco_results.py \
  --dataset-path datasets/coco/images/val2017/ \
  --gt-annotations datasets/coco/annotations/instances_val2017.json \
  --weights weights/model_n.pt \
  --output-json results/baseline_preds.json \
  --method baseline \
  --device mps
```

**Evaluate with pycocotools**:
```bash
python scripts/evaluate_coco.py \
  --annotation-file datasets/coco/annotations/instances_val2017.json \
  --results-file results/adaptivision_preds.json
```

### Using as Python Library

```python
from src.adaptivision import AdaptiVision

detector = AdaptiVision(
    model_path='weights/model_n.pt',
    device='auto',
    conf_threshold=0.25,
    iou_threshold=0.45,
    enable_adaptive_confidence=True,
    context_aware=True,
    enable_postprocess_filter=True
)

# Detect objects
results = detector.predict('path/to/image.jpg')
detection_data = results[0]

# Visualize
detector.visualize('path/to/image.jpg', detection_data, 'output.jpg')
```

## Project Structure

```
AdaptiVision/
├── src/
│   ├── adaptivision.py       # Core AdaptiVision class
│   ├── cli.py                 # CLI interface
│   ├── compare_methods.py     # Comparison logic
│   ├── create_visualizations.py  # Visualization tools
│   └── utils.py               # Shared utilities
├── scripts/
│   ├── run_experiments.py     # Full experimental pipeline
│   ├── save_coco_results.py   # Generate COCO predictions
│   ├── evaluate_coco.py       # Calculate mAP metrics
│   ├── generate_capped_time_plot.py    # Plotting utilities
│   └── generate_overhead_plot.py       # Plotting utilities
├── examples/
│   ├── basic_detection.py     # Simple detection example
│   └── batch_processing.py    # Batch processing example
├── datasets/                  # Dataset storage (not included)
├── weights/                   # Model weights
│   └── model_n.pt            # YOLOv8 nano model
├── samples/                   # Sample images
└── research_paper/            # Research paper and figures
```

## Important Implementation Details

### Confidence Threshold Tuning
- Base threshold: 0.25 (configurable)
- Initial detection pass: 0.05 (to capture potential objects)
- Adaptive range: 0.10 to 0.95 (clamped after class adjustments)
- Context boost: +0.02 for objects near related objects
- Class adjustments range: -0.05 (small objects) to +0.05 (large, clear objects)

### Scene Complexity Calculation
Weighted factors (sum to 1.0):
- `num_objects`: 0.5 (more objects = higher complexity)
- `object_size_var`: 0.25 (high size variance = higher complexity)
- `object_density`: 0.25 (higher density = higher complexity)

### Critical Dependencies
- `ultralytics==8.3.107` (specific version tested)
- `torch>=1.10.0` with appropriate backend (CUDA/MPS/CPU)
- `opencv-python>=4.5.0`
- `pycocotools` (for COCO evaluation)

### Testing Strategy
- No formal test suite currently
- Use `scripts/run_experiments.py` for validation
- Compare against baseline YOLO for regression testing
- Verify on COCO128 or COCO val2017 datasets

## Known Considerations

1. **Image Loading Robustness**: The system has PIL fallback for corrupted images that cv2.imread can't read
2. **Parallel Processing**: Use `--workers` parameter for batch processing to leverage multiple cores
3. **Memory**: Large datasets benefit from batch processing with controlled worker count
4. **YOLO Model**: Works with YOLOv8 models from Ultralytics; newer versions may introduce breaking changes
5. **Evaluation Methods**: Two approaches available:
   - `run_experiments.py`: Comparative analysis with visualizations
   - `save_coco_results.py` + `evaluate_coco.py`: Standard COCO mAP evaluation

## Entry Points

**After installation** (`pip install -e .`):
```bash
adaptivision detect --image <path> --output <path>
```

**Direct execution**:
```bash
python src/cli.py <subcommand> [options]
python scripts/run_experiments.py [options]
python scripts/save_coco_results.py [options]
```

## Documentation References

- Main documentation: [README.md](README.md)
- Script usage details: [SCRIPT_USAGE.md](SCRIPT_USAGE.md)
- Research paper: [research_paper/adaptivision_paper.pdf](research_paper/adaptivision_paper.pdf)
