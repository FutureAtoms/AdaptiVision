# ============================================================================
# AdaptiVision - Reproduce Paper Results (Windows PowerShell)
# ============================================================================
#
# This script reproduces the experimental results from the AdaptiVision paper.
# It's designed for complete beginners - just right-click and "Run with PowerShell"!
#
# Prerequisites:
#   - Windows 10/11
#   - Python 3.8 or higher (will check and guide you)
#   - Internet connection (to download dependencies)
#
# What this script does:
#   1. Checks Python installation
#   2. Creates virtual environment
#   3. Installs all dependencies
#   4. Downloads model weights
#   5. Downloads sample dataset (COCO128)
#   6. Runs the paper's experiments
#   7. Generates all figures and results
#
# Estimated time: 15-30 minutes (depending on internet speed)
# Disk space needed: ~500 MB
# ============================================================================

# Set strict mode
$ErrorActionPreference = "Stop"

# Color functions
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Step {
    param([string]$Message)
    Write-ColorOutput "`n$Message" "Cyan"
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✓ $Message" "Green"
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput "✗ $Message" "Red"
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠ $Message" "Yellow"
}

# Header
Clear-Host
Write-ColorOutput "============================================================================" "Cyan"
Write-ColorOutput "                   AdaptiVision Paper Reproduction" "Cyan"
Write-ColorOutput "============================================================================" "Cyan"
Write-Host ""
Write-Host "This script will reproduce all experiments from the AdaptiVision paper."
Write-Host ""
Write-Warning "What will be installed:"
Write-Host "  - Python virtual environment"
Write-Host "  - Required Python packages (PyTorch, OpenCV, etc.)"
Write-Host "  - YOLOv8 model weights (~6 MB)"
Write-Host "  - COCO128 sample dataset (~100 MB)"
Write-Host ""
Write-Warning "What will be generated:"
Write-Host "  - Experimental results (detection images)"
Write-Host "  - Performance comparisons"
Write-Host "  - Visualizations (complexity maps, threshold maps)"
Write-Host "  - Summary statistics and plots"
Write-Host "  - Experiment report (Markdown)"
Write-Host ""
Write-Warning "Estimated time: 15-30 minutes"
Write-Warning "Disk space needed: ~500 MB"
Write-Host ""
$confirm = Read-Host "Press Enter to continue or Ctrl+C to cancel"

# ============================================================================
# Step 1: Check Python Installation
# ============================================================================

Write-Step "[Step 1/7] Checking Python installation..."

try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }

    Write-Host "Found: $pythonVersion"

    # Extract version numbers
    if ($pythonVersion -match "Python (\d+)\.(\d+)\.(\d+)") {
        $major = [int]$matches[1]
        $minor = [int]$matches[2]

        if (($major -lt 3) -or ($major -eq 3 -and $minor -lt 8)) {
            Write-Error "Python 3.8+ required, found $pythonVersion"
            Write-Host "Please install Python 3.8 or higher from: https://www.python.org/downloads/"
            exit 1
        }
    }

    Write-Success "Python $pythonVersion detected (OK)"
}
catch {
    Write-Error "Python is not installed or not in PATH"
    Write-Host ""
    Write-Host "Please install Python 3.8 or higher from:"
    Write-Host "https://www.python.org/downloads/"
    Write-Host ""
    Write-Warning "Important: During installation, check 'Add Python to PATH'"
    exit 1
}

# ============================================================================
# Step 2: Create Virtual Environment
# ============================================================================

Write-Step "[Step 2/7] Creating virtual environment..."

if (Test-Path "venv") {
    Write-Host "Virtual environment already exists."
    $recreate = Read-Host "Do you want to recreate it? (y/N)"
    if ($recreate -eq "y" -or $recreate -eq "Y") {
        Write-Host "Removing old virtual environment..."
        Remove-Item -Recurse -Force venv
    }
    else {
        Write-Host "Using existing virtual environment..."
        goto SkipVenvCreation
    }
}

Write-Host "Creating new virtual environment..."
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment"
    exit 1
}

Write-Success "Virtual environment created"

:SkipVenvCreation

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to activate virtual environment"
    Write-Host ""
    Write-Warning "If you see an execution policy error, run PowerShell as Administrator and execute:"
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}

Write-Success "Virtual environment activated"

# ============================================================================
# Step 3: Install Dependencies
# ============================================================================

Write-Step "[Step 3/7] Installing dependencies..."
Write-Host "This may take 5-10 minutes depending on your internet connection..."
Write-Host ""

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install package
Write-Host "Installing AdaptiVision package..."
pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install AdaptiVision package"
    exit 1
}

Write-Host "Installing Ultralytics 8.3.107..."
pip install ultralytics==8.3.107 --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install Ultralytics"
    exit 1
}

Write-Host "Installing additional dependencies..."
pip install pycocotools --quiet 2>$null

Write-Success "All dependencies installed"

# ============================================================================
# Step 4: Download Model Weights
# ============================================================================

Write-Step "[Step 4/7] Downloading model weights..."

if (Test-Path "weights\model_n.pt") {
    Write-Host "Model weights already exist."
    $redownload = Read-Host "Do you want to re-download? (y/N)"
    if ($redownload -ne "y" -and $redownload -ne "Y") {
        goto SkipWeights
    }
}

Write-Host "Downloading YOLOv8 nano model (~6 MB)..."
python scripts\download_weights.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download model weights"
    exit 1
}

:SkipWeights
Write-Success "Model weights ready"

# ============================================================================
# Step 5: Download COCO128 Dataset
# ============================================================================

Write-Step "[Step 5/7] Downloading COCO128 dataset..."

if (Test-Path "datasets\coco128\images\train2017") {
    Write-Host "COCO128 dataset already exists."
    $redownload = Read-Host "Do you want to re-download? (y/N)"
    if ($redownload -ne "y" -and $redownload -ne "Y") {
        goto SkipDataset
    }
    Remove-Item -Recurse -Force datasets\coco128
}

Write-Host ""
Write-Host "Downloading COCO128 dataset (~100 MB)..."
Write-Host "This is a sample of 128 images from the COCO dataset."
Write-Host ""

# Create datasets directory
if (-not (Test-Path "datasets")) {
    New-Item -ItemType Directory -Force -Path datasets | Out-Null
}

# Download using Python
$downloadScript = @"
from ultralytics.data.utils import download
download(['https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'], dir='datasets')
"@

python -c $downloadScript
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to download COCO128 dataset"
    Write-Host ""
    Write-Host "You can manually download it from:"
    Write-Host "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    Write-Host ""
    Write-Host "Extract to: datasets\coco128\"
    exit 1
}

:SkipDataset
Write-Success "COCO128 dataset ready"

# ============================================================================
# Step 6: Run Smoke Test
# ============================================================================

Write-Step "[Step 6/7] Running smoke test..."
Write-Host "This quick test ensures everything is working correctly."
Write-Host ""

python smoke_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Smoke test failed"
    Write-Host "Please check the error messages above."
    exit 1
}

Write-Success "Smoke test passed"

# ============================================================================
# Step 7: Run Paper Experiments
# ============================================================================

Write-Step "[Step 7/7] Running paper experiments..."
Write-Host ""
Write-Host "This will reproduce the results from the AdaptiVision paper:"
Write-Host "  - Standard YOLO detection on COCO128"
Write-Host "  - AdaptiVision detection on COCO128"
Write-Host "  - Side-by-side comparisons"
Write-Host "  - Complexity visualizations"
Write-Host "  - Performance analytics"
Write-Host "  - Summary statistics"
Write-Host ""
Write-Warning "This may take 10-15 minutes..."
Write-Host ""
$confirm = Read-Host "Press Enter to start experiments"

# Create output directory with timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$expDir = "results\paper_reproduction_$timestamp"

Write-Host "Output directory: $expDir"
Write-Host ""

# Run experiments
python scripts\run_experiments.py --data datasets\coco128\images\train2017 --output $expDir --weights weights\model_n.pt --device cpu

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "Experiments failed"
    Write-Host "Please check the error messages above."
    exit 1
}

Write-Host ""
Write-Success "Experiments completed successfully!"

# ============================================================================
# Results Summary
# ============================================================================

Write-Host ""
Write-ColorOutput "============================================================================" "Cyan"
Write-ColorOutput "                          RESULTS SUMMARY" "Cyan"
Write-ColorOutput "============================================================================" "Cyan"
Write-Host ""
Write-Host "Experiments completed successfully!"
Write-Host ""
Write-Warning "Results location: $expDir"
Write-Host ""
Write-Warning "Generated files:"
Write-Host "  📁 standard/           - Standard YOLO detection results"
Write-Host "  📁 adaptive/           - AdaptiVision detection results"
Write-Host "  📁 comparisons/        - Side-by-side comparison images"
Write-Host "  📁 visualizations/     - Complexity and threshold maps"
Write-Host "  📁 analytics/          - Performance plots and statistics"
Write-Host "  📄 experiment_report.md - Detailed experiment report"
Write-Host "  📄 summary_results.csv - Tabular results"
Write-Host "  📄 detailed_results.json - Raw experimental data"
Write-Host ""

# Show summary if available
if (Test-Path "$expDir\experiment_report.md") {
    Write-Warning "Quick Summary:"
    Write-Host ""
    Get-Content "$expDir\experiment_report.md" | Select-String "Total images|Standard detection|Adaptive detection|Average speedup"
    Write-Host ""
}

Write-ColorOutput "To view results:" "Green"
Write-Host "  1. Open: $expDir\experiment_report.md"
Write-Host "  2. Browse images in: $expDir\comparisons\"
Write-Host "  3. Check analytics: $expDir\analytics\"
Write-Host ""

Write-Warning "Key findings from the paper (expected):"
Write-Host "  • AdaptiVision is 6-9x faster than standard YOLO"
Write-Host "  • Detects 25% more objects overall"
Write-Host "  • 2x better at detecting small objects (books, phones)"
Write-Host "  • Adaptive thresholds reduce false positives"
Write-Host ""

# Ask to open results
$openResults = Read-Host "Would you like to open the results folder now? (Y/n)"
if ($openResults -ne "n" -and $openResults -ne "N") {
    explorer $expDir
}

Write-Host ""
Write-ColorOutput "============================================================================" "Cyan"
Write-ColorOutput "                   PAPER REPRODUCTION COMPLETE!" "Cyan"
Write-ColorOutput "============================================================================" "Cyan"
Write-Host ""
Write-Success "Thank you for reproducing the AdaptiVision paper!"
Write-Host ""
Write-Host "If you use these results, please cite:"
Write-Host "  AdaptiVision: Adaptive Context-Aware Object Detection"
Write-Host "  Abhilash Chadhar, 2025"
Write-Host ""
Write-Host "For more information:"
Write-Host "  📖 Paper: research_paper\adaptivision_paper.pdf"
Write-Host "  🌐 GitHub: https://github.com/FutureAtoms/AdaptiVision"
Write-Host "  📧 Contact: contact@future-mind.org"
Write-Host ""
Write-Host "To run experiments again: .\reproduce_paper_windows.ps1"
Write-Host "To run on different images: python scripts\run_experiments.py --help"
Write-Host ""
Read-Host "Press Enter to exit"
