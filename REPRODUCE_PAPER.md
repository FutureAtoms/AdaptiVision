# Reproducing the AdaptiVision Paper Results

This guide explains how to reproduce all experimental results from the AdaptiVision research paper.

## Quick Start (Windows Users)

### Option 1: Batch Script (Easiest)

1. **Download the repository:**
   - Go to https://github.com/FutureAtoms/AdaptiVision
   - Click "Code" → "Download ZIP"
   - Extract to a folder (e.g., `C:\AdaptiVision`)

2. **Run the reproduction script:**
   - Double-click: `reproduce_paper_windows.bat`
   - Follow the prompts
   - Wait 15-30 minutes
   - Done!

### Option 2: PowerShell Script (Recommended for Windows 10/11)

1. **Download the repository** (same as above)

2. **Run PowerShell as Administrator:**
   - Right-click on PowerShell
   - Select "Run as Administrator"

3. **Set execution policy (first time only):**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Navigate to the project:**
   ```powershell
   cd C:\AdaptiVision
   ```

5. **Run the script:**
   ```powershell
   .\reproduce_paper_windows.ps1
   ```

6. **Follow the prompts**
   - Total time: 15-30 minutes
   - Results appear in `results\paper_reproduction_*\`

## What Gets Reproduced

The reproduction scripts generate ALL results from the paper:

### 1. Detection Results

**Standard YOLO:**
- Runs YOLOv8 with fixed confidence threshold (0.25)
- Processes all 128 COCO128 images
- Saves annotated images to `results/*/standard/`

**AdaptiVision:**
- Runs adaptive threshold detection
- Includes context-aware reasoning
- Includes geometric post-processing
- Saves annotated images to `results/*/adaptive/`

### 2. Comparison Images

Side-by-side comparisons showing:
- Standard detection (left)
- Adaptive detection (right)
- Saved to `results/*/comparisons/`

### 3. Visualizations

For each image:
- **Complexity Map**: Visual representation of scene complexity
- **Threshold Map**: Adaptive threshold values per region
- Saved to `results/*/visualizations/<image_name>/`

### 4. Performance Analytics

**Plots generated:**
- Detection time comparison (standard vs adaptive)
- Object count comparison
- Speedup distribution
- Scene complexity distribution
- Saved to `results/*/analytics/`

**Tables generated:**
- `summary_results.csv`: Per-image results
- `detailed_results.json`: Raw experimental data

### 5. Experiment Report

Comprehensive Markdown report including:
- Summary statistics
- Performance metrics
- Key findings
- Saved as `experiment_report.md`

## Prerequisites

### Windows

**Required:**
- Windows 10 or Windows 11
- Python 3.8 or higher ([Download](https://www.python.org/downloads/))
  - ⚠️ **Important**: Check "Add Python to PATH" during installation
- Internet connection (for downloading dependencies)
- ~500 MB free disk space

**Optional but Recommended:**
- NVIDIA GPU with CUDA (for faster processing)
- 8GB+ RAM (16GB recommended)

### Installing Python (if not installed)

1. Go to https://www.python.org/downloads/
2. Download Python 3.11 (latest stable)
3. Run installer
4. ✅ **Check "Add Python to PATH"**
5. Click "Install Now"
6. Verify installation:
   ```cmd
   python --version
   ```

## Step-by-Step Manual Reproduction

If you prefer to run steps manually or are troubleshooting:

### Step 1: Clone Repository

```bash
git clone https://github.com/FutureAtoms/AdaptiVision.git
cd AdaptiVision
```

Or download ZIP and extract.

### Step 2: Create Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -e .
pip install ultralytics==8.3.107
```

### Step 4: Download Model Weights

```bash
python scripts\download_weights.py
```

### Step 5: Download COCO128 Dataset

The COCO128 dataset will be automatically downloaded when you run experiments. Alternatively, download manually:

1. Download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
2. Extract to: `datasets\coco128\`
3. Verify structure:
   ```
   datasets/
   └── coco128/
       └── images/
           └── train2017/
               ├── 000000000009.jpg
               ├── 000000000025.jpg
               └── ... (128 images total)
   ```

### Step 6: Run Smoke Test

Verify everything works:

```bash
python smoke_test.py
```

Expected output:
```
AdaptiVision Smoke Test
============================================================
Detected 6 objects
SUCCESS! Smoke test passed.
```

### Step 7: Run Experiments

**Full experiment (128 images, ~15 minutes):**
```bash
python scripts\run_experiments.py ^
  --data datasets\coco128\images\train2017 ^
  --output results\my_experiment ^
  --weights weights\model_n.pt ^
  --device cpu
```

**Quick test (3 images, ~2 minutes):**
```bash
python scripts\run_experiments.py ^
  --data samples\coco ^
  --output results\quick_test ^
  --weights weights\model_n.pt ^
  --device cpu
```

**With GPU (if available):**
```bash
python scripts\run_experiments.py ^
  --data datasets\coco128\images\train2017 ^
  --output results\my_experiment_gpu ^
  --weights weights\model_n.pt ^
  --device cuda
```

## Expected Results

### Performance Metrics (from the paper)

| Metric | Standard YOLO | AdaptiVision | Improvement |
|--------|--------------|--------------|-------------|
| **Average Processing Time** | 0.145s | 0.024s | **6.0× faster** |
| **Total Objects Detected** | 1,064 | 1,334 | **+25.4%** |
| **Small Objects** | 131 | 281 | **+114.5%** |

### Class-Specific Results

| Class | Standard | Adaptive | Improvement |
|-------|----------|----------|-------------|
| person | 217 | 283 | +30.4% |
| book | 12 | 29 | +141.7% |
| bottle | 43 | 62 | +44.2% |
| cell phone | 5 | 12 | +140.0% |
| cup | 21 | 28 | +33.3% |

### Your Results

After running experiments, check:

1. **Experiment Report**: `results\*/experiment_report.md`
   - Overall statistics
   - Performance summary
   - Key findings

2. **Summary Table**: `results\*/summary_results.csv`
   - Open in Excel/LibreOffice
   - Per-image metrics
   - Easy to analyze

3. **Visual Results**: `results\*/comparisons\`
   - Browse comparison images
   - See side-by-side differences
   - Verify improvements

## Troubleshooting

### Python Not Found

**Error:**
```
'python' is not recognized as an internal or external command
```

**Solution:**
1. Reinstall Python
2. ✅ Check "Add Python to PATH"
3. Restart Command Prompt/PowerShell

### PowerShell Execution Policy

**Error:**
```
cannot be loaded because running scripts is disabled on this system
```

**Solution:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Virtual Environment Activation Failed

**Error:**
```
Activate.ps1 cannot be loaded
```

**Solution:**
Use Command Prompt instead of PowerShell, or fix execution policy (see above).

### COCO128 Download Failed

**Solution:**
Manually download and extract:
1. Download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
2. Extract to `datasets\coco128\`

### Out of Memory

**Error:**
```
CUDA out of memory
```

**Solution:**
```bash
# Use CPU instead
python scripts\run_experiments.py --device cpu ...
```

### Import Errors

**Error:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solution:**
Make sure virtual environment is activated:
```cmd
venv\Scripts\activate
pip install -e .
```

## Customizing Experiments

### Run on Your Own Images

```bash
python scripts\run_experiments.py ^
  --data path\to\your\images ^
  --output results\custom_experiment ^
  --weights weights\model_n.pt ^
  --device cpu
```

### Adjust Confidence Threshold

Edit `scripts\run_experiments.py`:
```python
# Line ~150
conf_threshold=0.25  # Change to 0.20 or 0.30
```

### Use Different Model

Download larger model:
```bash
python scripts\download_weights.py --model yolov8s  # Small model
python scripts\download_weights.py --model yolov8m  # Medium model
```

Use in experiments:
```bash
python scripts\run_experiments.py --weights weights\model_s.pt ...
```

## Advanced: Full COCO Validation Set

For complete paper reproduction with full COCO validation (5000 images):

### 1. Download COCO Val2017

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path datasets\coco

# Download images (~1GB)
Invoke-WebRequest -Uri http://images.cocodataset.org/zips/val2017.zip -OutFile datasets\val2017.zip

# Download annotations
Invoke-WebRequest -Uri http://images.cocodataset.org/annotations/annotations_trainval2017.zip -OutFile datasets\annotations.zip

# Extract
Expand-Archive datasets\val2017.zip -DestinationPath datasets\coco\images\
Expand-Archive datasets\annotations.zip -DestinationPath datasets\coco\
```

### 2. Run Full Evaluation

```bash
# Generate predictions (takes ~2-3 hours on CPU)
python scripts\save_coco_results.py ^
  --dataset-path datasets\coco\images\val2017 ^
  --gt-annotations datasets\coco\annotations\instances_val2017.json ^
  --weights weights\model_n.pt ^
  --output-json results\adaptivision_full_preds.json ^
  --method adaptivision ^
  --device cpu

# Calculate mAP
python scripts\evaluate_coco.py ^
  --annotation-file datasets\coco\annotations\instances_val2017.json ^
  --results-file results\adaptivision_full_preds.json
```

## Citing This Work

If you use these reproduction scripts or results, please cite:

```bibtex
@article{adaptivision2025,
  title={AdaptiVision: Adaptive Context-Aware Object Detection},
  author={Chadhar, Abhilash},
  year={2025},
  journal={arXiv preprint},
  note={Code available at https://github.com/FutureAtoms/AdaptiVision}
}
```

## Support

- **GitHub Issues**: https://github.com/FutureAtoms/AdaptiVision/issues
- **Paper**: `research_paper/adaptivision_paper.pdf`
- **Email**: contact@future-mind.org
- **Documentation**: See README.md, SCRIPT_USAGE.md

## Verification Checklist

After running reproduction scripts:

- [ ] Smoke test passed
- [ ] Experiment completed without errors
- [ ] `experiment_report.md` generated
- [ ] Comparison images show differences
- [ ] Performance matches paper (~6x speedup)
- [ ] Small objects detected better (books, phones)
- [ ] Results folder contains all outputs

## Next Steps

After reproducing the paper:

1. **Explore Results**
   - Browse comparison images
   - Read experiment report
   - Analyze CSV data in Excel

2. **Modify Experiments**
   - Try different confidence thresholds
   - Test on your own images
   - Compare different model sizes

3. **Contribute**
   - Share your results
   - Report bugs/issues
   - Suggest improvements

4. **Build On It**
   - Use AdaptiVision in your project
   - Extend the methodology
   - Publish your findings

---

**Questions?** Open an issue on GitHub or contact us!
