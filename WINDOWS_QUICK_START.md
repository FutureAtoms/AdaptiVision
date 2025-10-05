# Windows Quick Start Guide

**For your sister or any Windows beginner who wants to reproduce the paper**

## The Absolute Easiest Way

### 1. Download the Repository

Click this link: https://github.com/FutureAtoms/AdaptiVision/archive/refs/heads/main.zip

Save it to your Downloads folder, then:
- Right-click the ZIP file
- Click "Extract All..."
- Choose a location (like `C:\AdaptiVision`)
- Click "Extract"

### 2. Run the Magic Script

Navigate to the folder you just extracted, then:

**Double-click:** `reproduce_paper_windows.bat`

That's it! The script will:
- Check if Python is installed (and tell you how to install it if not)
- Set up everything automatically
- Download the dataset
- Run all experiments
- Generate all results

**Time:** 15-30 minutes
**Result:** Complete paper reproduction in `results\paper_reproduction_*\`

## What if Python isn't installed?

The script will tell you! If you see an error about Python:

1. Go to: https://www.python.org/downloads/
2. Click the big yellow "Download Python" button
3. Run the installer
4. ⚠️ **IMPORTANT:** Check the box "Add Python to PATH"
5. Click "Install Now"
6. When done, run `reproduce_paper_windows.bat` again

## What Will You Get?

After the script finishes, you'll have a folder like:
`results\paper_reproduction_20251005_143022\`

Inside, you'll find:

### 📁 **comparisons/**
Side-by-side images showing:
- Left side: Standard YOLO detection
- Right side: AdaptiVision detection
- You can visually see the improvements!

**Example:** Open `comparison_000000000009.jpg` and you'll see AdaptiVision detects more objects.

### 📁 **visualizations/**
Cool heatmaps showing:
- How complex each scene is (red = complex, blue = simple)
- What threshold AdaptiVision chose for each area
- Why it made those choices

### 📁 **analytics/**
Graphs and charts:
- `detection_time_comparison.png` - Shows AdaptiVision is ~6x faster
- `object_count_comparison.png` - Shows it detects ~25% more objects
- `speedup_distribution.png` - How much faster across all images

### 📄 **experiment_report.md**
Open this in Notepad or any text editor. It's a complete summary:
- How many objects were detected
- How fast each method was
- Which method performed better

### 📄 **summary_results.csv**
Open in Excel or Google Sheets:
- One row per image
- Columns for time, objects detected, speedup
- Easy to analyze and make your own charts

## Troubleshooting

### "Python is not recognized"

**You need to install Python.**
Download from: https://www.python.org/downloads/
⚠️ Check "Add Python to PATH" during installation!

### "Running scripts is disabled"

**You're using PowerShell and it's blocked.**
Option 1: Use the `.bat` file instead (double-click `reproduce_paper_windows.bat`)
Option 2: Open PowerShell as Administrator and run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Script seems stuck

**It's probably downloading something.**
The script downloads ~100MB of data the first time:
- Model weights: ~6 MB (quick)
- COCO128 dataset: ~100 MB (takes a few minutes)

Just wait! You'll see progress messages.

### Not enough disk space

**You need ~500 MB free.**
- Model weights: 6 MB
- Dataset: ~100 MB
- Results: ~300 MB
- Python packages: ~100 MB

## Advanced: Run Experiments on Your Own Images

1. Put your images in a folder (e.g., `C:\MyImages\`)

2. Open Command Prompt (`cmd`)

3. Navigate to AdaptiVision folder:
   ```cmd
   cd C:\AdaptiVision
   ```

4. Activate virtual environment:
   ```cmd
   venv\Scripts\activate
   ```

5. Run experiments:
   ```cmd
   python scripts\run_experiments.py --data C:\MyImages --output results\my_custom_experiment --weights weights\model_n.pt
   ```

6. Check results in `results\my_custom_experiment\`

## What the Paper Says You Should See

From the AdaptiVision paper, you should expect:

✓ **Speed:** AdaptiVision is 6-9x faster than standard YOLO
✓ **Accuracy:** Detects 25% more objects overall
✓ **Small Objects:** 2x better at detecting books, phones, cups
✓ **Smart:** Adjusts threshold based on scene complexity

Your results should be similar! Small variations are normal due to:
- Different CPU/GPU
- Different Python version
- Random initialization

## Files Explained

| File | What It Does |
|------|--------------|
| `reproduce_paper_windows.bat` | Batch script (works on all Windows) |
| `reproduce_paper_windows.ps1` | PowerShell script (modern Windows 10/11) |
| `smoke_test.py` | Quick test to verify installation |
| `scripts\download_weights.py` | Downloads AI model weights |
| `scripts\run_experiments.py` | Main experiment runner |

## Need Help?

1. **Read the error message** - It usually tells you what's wrong!
2. **Check REPRODUCE_PAPER.md** - More detailed troubleshooting
3. **GitHub Issues:** https://github.com/FutureAtoms/AdaptiVision/issues
4. **Email:** contact@future-mind.org

## Success Checklist

After running, verify:

- [ ] Script completed without errors
- [ ] `experiment_report.md` exists in results folder
- [ ] Comparison images show detections
- [ ] Analytics folder has graphs
- [ ] CSV file can be opened in Excel
- [ ] AdaptiVision is ~6x faster (check report)
- [ ] More objects detected with AdaptiVision

If all checked ✓ - **Congratulations!** You've successfully reproduced the paper!

## Next Steps

1. **Explore Results**
   - Browse comparison images
   - Open experiment report
   - Look at analytics graphs

2. **Share Results**
   - Take screenshots of interesting comparisons
   - Share speedup numbers
   - Compare with paper results

3. **Learn More**
   - Read the full paper: `research_paper\adaptivision_paper.pdf`
   - Try different images
   - Experiment with settings

4. **Cite If You Use It**
   ```
   AdaptiVision: Adaptive Context-Aware Object Detection
   Abhilash Chadhar, 2025
   GitHub: https://github.com/FutureAtoms/AdaptiVision
   ```

## Summary

**Easiest Way:**
1. Download ZIP from GitHub
2. Extract to folder
3. Double-click `reproduce_paper_windows.bat`
4. Wait 15-30 minutes
5. Check results folder
6. Done! 🎉

**What You Get:**
- All paper results reproduced
- Visual comparisons
- Performance graphs
- Detailed statistics
- Experiment report

**No Coding Required!**

---

*Made with ❤️ for beginners - if you have any issues, please open a GitHub issue!*
