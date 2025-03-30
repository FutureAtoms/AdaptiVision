# Measurement Verification

This document provides a detailed breakdown of how key metrics and measurements were derived in the AdaptiVision experiments.

## Data Collection Methodology

All metrics are collected directly during the object detection process, with no simulated or fabricated data. The following measurements are recorded for each image processed:

1. **Standard Detection**:
   - Objects detected (count and classes)
   - Inference time
   - Confidence scores

2. **Adaptive Detection**:
   - Objects detected (count and classes)
   - Inference time
   - Confidence scores
   - Scene complexity score
   - Adaptive threshold used

## Performance Metrics Calculation

### Speed Improvement

Speed improvement is calculated as the ratio of standard detection time to adaptive detection time:

```python
speed_improvement = standard_detection_time / adaptive_detection_time
```

Example from image `000000000629.jpg`:
- Standard detection time: 0.0321s
- Adaptive detection time: 0.0142s
- Speed improvement: 0.0321/0.0142 = 2.26×

### Detection Performance

Detection performance is measured by comparing the object count between standard and adaptive methods:

```python
detection_improvement = (adaptive_count - standard_count) / standard_count * 100
```

Example from image `000000000009.jpg`:
- Standard detection: 4 objects
- Adaptive detection: 6 objects
- Improvement: (6-4)/4 * 100 = 50%

### Scene Complexity

Scene complexity is calculated using a weighted combination of:
- Initial object count (weight: 0.4)
- Object size variance (weight: 0.3) 
- Spatial density (weight: 0.3)

The final score is normalized to a range of 0.0-1.0.

Example calculation:
```python
def calculate_complexity(detections):
    if not detections:
        return 0.0
    
    # Count factor (more objects = higher complexity)
    count = len(detections)
    count_factor = min(1.0, count / 30)  # Normalize with max of 30 objects
    
    # Size variance factor
    sizes = [d.bbox_area for d in detections]
    size_variance = np.var(sizes) if len(sizes) > 1 else 0
    size_factor = min(1.0, size_variance / 10000)
    
    # Spatial distribution factor
    positions = [d.bbox_center for d in detections]
    if len(positions) > 1:
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                distances.append(euclidean_distance(positions[i], positions[j]))
        avg_distance = np.mean(distances)
        spatial_factor = 1.0 - min(1.0, avg_distance / 500)  # Closer objects = more complex
    else:
        spatial_factor = 0.0
    
    # Weighted combination
    complexity = (0.4 * count_factor) + (0.3 * size_factor) + (0.3 * spatial_factor)
    return complexity
```

## Verification Process

For each metric, we employ the following validation steps:

1. **Raw Data Preservation**: All raw measurements are stored in JSON format for each image
2. **Statistical Analysis**: Summary statistics are computed from raw data without filtering
3. **Visualization**: Plots are generated directly from the collected data points
4. **Cross-Validation**: A random subset of images is manually verified

## Example Verification

For image `000000000575.jpg`:

```json
{
  "image_id": "000000000575",
  "standard_detection": {
    "object_count": 7,
    "classes_detected": ["person", "car", "chair", "bottle", "cup"],
    "processing_time": 0.0356
  },
  "adaptive_detection": {
    "object_count": 11,
    "classes_detected": ["person", "car", "chair", "bottle", "cup", "book"],
    "processing_time": 0.0148,
    "complexity_score": 0.72,
    "adaptive_threshold": 0.18
  },
  "metrics": {
    "speed_improvement": 2.41,
    "detection_improvement": 57.14,
    "threshold_reduction": 0.07
  }
}
```

## Reproduction Steps

To verify any result:

1. Run the following command to process a specific image:
   ```
   python src/cli.py compare <image_path> --save-metrics
   ```

2. Examine the generated metrics JSON file in `results/metrics/`.

3. For aggregate statistics, run:
   ```
   python scripts/analyze_results.py
   ```

All measurement code can be found in the following files:
- `src/adaptive/metrics.py`: Metrics calculation functions
- `src/adaptive/complexity.py`: Scene complexity analysis
- `scripts/analyze_results.py`: Statistical analysis

## Experimental Controls

To ensure fair comparison:
- Both detection methods use the same base YOLOv8 model
- Processing is done on the same hardware for both methods
- Images are processed in the same resolution
- The same image preprocessing steps are applied
- All tests are run with identical batch sizes 