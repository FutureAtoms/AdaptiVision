@echo off
REM ============================================================================
REM AdaptiVision - Reproduce Paper Results (Windows)
REM ============================================================================
REM
REM This script reproduces the experimental results from the AdaptiVision paper.
REM It's designed for complete beginners - just double-click and follow prompts!
REM
REM Prerequisites:
REM   - Windows 10/11
REM   - Python 3.8 or higher (will check and guide you)
REM   - Internet connection (to download dependencies)
REM
REM What this script does:
REM   1. Checks Python installation
REM   2. Creates virtual environment
REM   3. Installs all dependencies
REM   4. Downloads model weights
REM   5. Downloads sample dataset (COCO128)
REM   6. Runs the paper's experiments
REM   7. Generates all figures and results
REM
REM Estimated time: 15-30 minutes (depending on internet speed)
REM Disk space needed: ~500 MB
REM ============================================================================

setlocal enabledelayedexpansion

REM Set colors for output
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo ============================================================================
echo                    AdaptiVision Paper Reproduction
echo ============================================================================
echo.
echo This script will reproduce all experiments from the AdaptiVision paper.
echo.
echo %YELLOW%What will be installed:%RESET%
echo   - Python virtual environment
echo   - Required Python packages (PyTorch, OpenCV, etc.)
echo   - YOLOv8 model weights (~6 MB)
echo   - COCO128 sample dataset (~100 MB)
echo.
echo %YELLOW%What will be generated:%RESET%
echo   - Experimental results (detection images)
echo   - Performance comparisons
echo   - Visualizations (complexity maps, threshold maps)
echo   - Summary statistics and plots
echo   - Experiment report (Markdown)
echo.
echo %YELLOW%Estimated time:%RESET% 15-30 minutes
echo %YELLOW%Disk space needed:%RESET% ~500 MB
echo.
pause

REM ============================================================================
REM Step 1: Check Python Installation
REM ============================================================================

echo.
echo %BLUE%[Step 1/7]%RESET% Checking Python installation...
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%ERROR: Python is not installed or not in PATH%RESET%
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo %YELLOW%Important:%RESET% During installation, check "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo Found Python version: %PYVER%

REM Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJOR=%%a
    set PYMINOR=%%b
)

if %PYMAJOR% LSS 3 (
    echo %RED%ERROR: Python 3.8+ required, found %PYVER%%RESET%
    pause
    exit /b 1
)

if %PYMAJOR% EQU 3 if %PYMINOR% LSS 8 (
    echo %RED%ERROR: Python 3.8+ required, found %PYVER%%RESET%
    pause
    exit /b 1
)

echo %GREEN%✓ Python %PYVER% detected (OK)%RESET%

REM ============================================================================
REM Step 2: Create Virtual Environment
REM ============================================================================

echo.
echo %BLUE%[Step 2/7]%RESET% Creating virtual environment...
echo.

if exist venv (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto skip_venv
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo %RED%ERROR: Failed to create virtual environment%RESET%
    pause
    exit /b 1
)

echo %GREEN%✓ Virtual environment created%RESET%

:skip_venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo %RED%ERROR: Failed to activate virtual environment%RESET%
    pause
    exit /b 1
)

echo %GREEN%✓ Virtual environment activated%RESET%

REM ============================================================================
REM Step 3: Install Dependencies
REM ============================================================================

echo.
echo %BLUE%[Step 3/7]%RESET% Installing dependencies...
echo.
echo This may take 5-10 minutes depending on your internet connection...
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo %YELLOW%Warning: Failed to upgrade pip, continuing...%RESET%
)

REM Install package
echo Installing AdaptiVision package...
pip install -e . --quiet
if errorlevel 1 (
    echo %RED%ERROR: Failed to install AdaptiVision package%RESET%
    pause
    exit /b 1
)

echo Installing Ultralytics 8.3.107...
pip install ultralytics==8.3.107 --quiet
if errorlevel 1 (
    echo %RED%ERROR: Failed to install Ultralytics%RESET%
    pause
    exit /b 1
)

echo Installing additional dependencies...
pip install pycocotools --quiet

echo %GREEN%✓ All dependencies installed%RESET%

REM ============================================================================
REM Step 4: Download Model Weights
REM ============================================================================

echo.
echo %BLUE%[Step 4/7]%RESET% Downloading model weights...
echo.

if exist weights\model_n.pt (
    echo Model weights already exist.
    choice /C YN /M "Do you want to re-download"
    if errorlevel 2 goto skip_weights
)

echo Downloading YOLOv8 nano model (~6 MB)...
python scripts\download_weights.py
if errorlevel 1 (
    echo %RED%ERROR: Failed to download model weights%RESET%
    pause
    exit /b 1
)

:skip_weights
echo %GREEN%✓ Model weights ready%RESET%

REM ============================================================================
REM Step 5: Download COCO128 Dataset
REM ============================================================================

echo.
echo %BLUE%[Step 5/7]%RESET% Downloading COCO128 dataset...
echo.

if exist datasets\coco128\images\train2017 (
    echo COCO128 dataset already exists.
    choice /C YN /M "Do you want to re-download"
    if errorlevel 2 goto skip_dataset
    rmdir /s /q datasets\coco128
)

echo.
echo Downloading COCO128 dataset (~100 MB)...
echo This is a sample of 128 images from the COCO dataset.
echo.

REM Create datasets directory
if not exist datasets mkdir datasets

REM Download using Python
python -c "from ultralytics.data.utils import download; download(['https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'], dir='datasets')"
if errorlevel 1 (
    echo %RED%ERROR: Failed to download COCO128 dataset%RESET%
    echo.
    echo You can manually download it from:
    echo https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
    echo.
    echo Extract to: datasets\coco128\
    echo.
    pause
    exit /b 1
)

:skip_dataset
echo %GREEN%✓ COCO128 dataset ready%RESET%

REM ============================================================================
REM Step 6: Run Smoke Test
REM ============================================================================

echo.
echo %BLUE%[Step 6/7]%RESET% Running smoke test...
echo.
echo This quick test ensures everything is working correctly.
echo.

python smoke_test.py
if errorlevel 1 (
    echo %RED%ERROR: Smoke test failed%RESET%
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo %GREEN%✓ Smoke test passed%RESET%

REM ============================================================================
REM Step 7: Run Paper Experiments
REM ============================================================================

echo.
echo %BLUE%[Step 7/7]%RESET% Running paper experiments...
echo.
echo This will reproduce the results from the AdaptiVision paper:
echo   - Standard YOLO detection on COCO128
echo   - AdaptiVision detection on COCO128
echo   - Side-by-side comparisons
echo   - Complexity visualizations
echo   - Performance analytics
echo   - Summary statistics
echo.
echo %YELLOW%This may take 10-15 minutes...%RESET%
echo.
pause

REM Create output directory with timestamp
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do (set mytime=%%a%%b)
set EXPDIR=results\paper_reproduction_%mydate%_%mytime%

echo Output directory: %EXPDIR%
echo.

REM Run experiments (use quotes to handle spaces and special chars)
python scripts\run_experiments.py --data datasets\coco128\images\train2017 --output "%EXPDIR%" --weights weights\model_n.pt --device cpu

if errorlevel 1 (
    echo.
    echo %RED%ERROR: Experiments failed%RESET%
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo %GREEN%✓ Experiments completed successfully!%RESET%

REM ============================================================================
REM Results Summary
REM ============================================================================

echo.
echo ============================================================================
echo                           RESULTS SUMMARY
echo ============================================================================
echo.
echo Experiments completed successfully!
echo.
echo %YELLOW%Results location:%RESET% %EXPDIR%
echo.
echo %YELLOW%Generated files:%RESET%
echo   📁 standard/           - Standard YOLO detection results
echo   📁 adaptive/           - AdaptiVision detection results
echo   📁 comparisons/        - Side-by-side comparison images
echo   📁 visualizations/     - Complexity and threshold maps
echo   📁 analytics/          - Performance plots and statistics
echo   📄 experiment_report.md - Detailed experiment report
echo   📄 summary_results.csv - Tabular results
echo   📄 detailed_results.json - Raw experimental data
echo.

REM Check if experiment report exists and show summary
if exist "%EXPDIR%\experiment_report.md" (
    echo %YELLOW%Quick Summary:%RESET%
    echo.
    findstr /C:"Total images" "%EXPDIR%\experiment_report.md"
    findstr /C:"Standard detection" "%EXPDIR%\experiment_report.md"
    findstr /C:"Adaptive detection" "%EXPDIR%\experiment_report.md"
    findstr /C:"Average speedup" "%EXPDIR%\experiment_report.md"
    echo.
)

echo %GREEN%To view results:%RESET%
echo   1. Open: %EXPDIR%\experiment_report.md
echo   2. Browse images in: %EXPDIR%\comparisons\
echo   3. Check analytics: %EXPDIR%\analytics\
echo.

echo %YELLOW%Key findings from the paper (expected):%RESET%
echo   • AdaptiVision is 6-9x faster than standard YOLO
echo   • Detects 25%% more objects overall
echo   • 2x better at detecting small objects (books, phones)
echo   • Adaptive thresholds reduce false positives
echo.

REM Ask to open results
choice /C YN /M "Would you like to open the results folder now"
if not errorlevel 2 (
    explorer "%EXPDIR%"
)

echo.
echo ============================================================================
echo                    PAPER REPRODUCTION COMPLETE!
echo ============================================================================
echo.
echo %GREEN%Thank you for reproducing the AdaptiVision paper!%RESET%
echo.
echo If you use these results, please cite:
echo   AdaptiVision: Adaptive Context-Aware Object Detection
echo   Abhilash Chadhar, 2025
echo.
echo For more information:
echo   📖 Paper: research_paper\adaptivision_paper.pdf
echo   🌐 GitHub: https://github.com/FutureAtoms/AdaptiVision
echo   📧 Contact: contact@future-mind.org
echo.
echo To run experiments again: reproduce_paper_windows.bat
echo To run on different images: python scripts\run_experiments.py --help
echo.
pause
