# Windows Setup Instructions

## Quick Smoke Test

### Step 1: Setup Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Dependencies
```cmd
pip install -e .
pip install ultralytics==8.3.107
```

### Step 3: Run Smoke Test
```cmd
python smoke_test.py
```

Expected output:
- Should detect 6 objects in bus.jpg
- Creates `results/smoke_test.jpg` with bounding boxes
- Takes about 5-10 seconds

## If You See Errors

### Error: "No module named 'cv2'"
**Solution**: Make sure virtual environment is activated
```cmd
venv\Scripts\activate
pip install opencv-python
```

### Error: "No such file or directory: 'weights\\model_n.pt'"
**Solution**: Download the model weights
```cmd
mkdir weights
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o weights/model_n.pt
```

Or manually download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
and place it in `weights/model_n.pt`

### Error: "No such file or directory: 'samples\\bus.jpg'"
**Solution**: The samples folder should be in the repo. Check you're in the right directory:
```cmd
dir samples
```

## Using the CLI (After Smoke Test Passes)

### Single Image Detection
```cmd
python src/cli.py detect --image samples/bus.jpg --output results/bus_detection.jpg --weights weights/model_n.pt --device auto
```

### Compare Adaptive vs Standard
```cmd
python src/cli.py compare --image samples/bus.jpg --output-dir results/comparisons/ --weights weights/model_n.pt
```

### Batch Processing
```cmd
python src/cli.py batch --input-dir samples/coco/ --output-dir results/batch/ --weights weights/model_n.pt --workers 2
```

## Device Selection

- `--device auto` (recommended): Automatically picks best device
- `--device cuda`: NVIDIA GPU (if you have one)
- `--device cpu`: CPU only (slower but always works)

## Common Windows Issues

### PowerShell Execution Policy
If you can't activate venv, run PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Path Too Long
If you get "path too long" errors, enable long paths in Windows 10/11:
1. Run `regedit` as Administrator
2. Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to 1

Or move the project folder closer to C:\ (e.g., `C:\AdaptiVision`)

### Backslash vs Forward Slash
The `smoke_test.py` script uses `pathlib` which handles both Windows and Unix paths automatically. This is why it works on both systems!
