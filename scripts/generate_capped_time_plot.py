# Script to generate a modified Figure A with capped Y-axis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Configuration ---
# Use the results from the successful reproduce_experiment run
RESULTS_CSV = Path("results/coco128_experiment/summary_results.csv")
OUTPUT_DIR = Path("results/coco128_experiment/analytics")
# Save with a new name to avoid overwriting the original
OUTPUT_FILENAME = "processing_time_comparison_capped.png" 
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME
Y_AXIS_LIMIT = 0.2 # Set the upper limit for the Y-axis (in seconds)

# --- Data Loading ---
try:
    df = pd.read_csv(RESULTS_CSV)
    print(f"Successfully loaded results from: {RESULTS_CSV}")
except FileNotFoundError:
    print(f"Error: Results file not found at {RESULTS_CSV}")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Plotting ---
plt.figure(figsize=(12, 7)) # Adjusted figure size for potentially many x-labels

# Create bar plot data
bar_width = 0.35
index = range(len(df))

plt.bar(index, df['Standard Time (s)'], bar_width, label='Standard Detection', alpha=0.8)
plt.bar([i + bar_width for i in index], df['Adaptive Time (s)'], bar_width, label='Adaptive Detection', alpha=0.8)

plt.xlabel('Images')
plt.ylabel('Processing Time (s)')
plt.title(f'Processing Time Comparison (Y-axis capped at {Y_AXIS_LIMIT}s)')
plt.xticks([i + bar_width / 2 for i in index], df['Filename'].str.replace('.jpg', '', regex=False), rotation=90, fontsize=8)
plt.ylim(0, Y_AXIS_LIMIT) # Apply the Y-axis limit
plt.legend()
plt.tight_layout() # Adjust layout to prevent labels overlapping
plt.grid(axis='y', linestyle='--', alpha=0.7)

# --- Save Figure ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH)
print(f"Capped processing time comparison plot saved to: {OUTPUT_PATH}")

# plt.show() # Uncomment if running interactively 