# Script to generate Figure B: Processing Time Overhead Histogram
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os # Import os

# --- Configuration ---
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__))) # Get script's directory
BASE_DIR = SCRIPT_DIR # In this case, script is in the base project dir

RESULTS_CSV = BASE_DIR / "results" / "coco128_full_experiment" / "summary_results.csv" # Build absolute path
OUTPUT_DIR = BASE_DIR / "results" / "coco128_full_experiment" / "analytics" # Build absolute path

# # RESULTS_CSV = Path("results/coco128_full_experiment/summary_results.csv") # Old relative path
# # OUTPUT_DIR = Path("results/coco128_full_experiment/analytics") # Old relative path

OUTPUT_FILENAME = "time_overhead_histogram.png" # Figure B
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_FILENAME

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

# --- Calculation ---
# Calculate the difference in milliseconds
df['Time Overhead (ms)'] = (df['Adaptive Time (s)'] - df['Standard Time (s)']) * 1000

# Exclude significant outliers for clearer visualization of the main distribution
# (Optional: Adjust percentile as needed, e.g., 99th or remove specific outliers)
q_low = df['Time Overhead (ms)'].quantile(0.01)
q_high = df['Time Overhead (ms)'].quantile(0.99)
df_filtered = df[(df['Time Overhead (ms)'] > q_low) & (df['Time Overhead (ms)'] < q_high)]

if df_filtered.empty:
    print("Warning: Filtering removed all data points. Plotting unfiltered data.")
    df_filtered = df # Fallback to unfiltered if filtering is too aggressive

print(f"Plotting distribution for {len(df_filtered)} images (outliers removed).")
median_overhead = df_filtered['Time Overhead (ms)'].median()
mean_overhead = df_filtered['Time Overhead (ms)'].mean()

# --- Plotting ---
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered['Time Overhead (ms)'], bins=20, kde=True)

plt.title('Distribution of Processing Time Overhead (Adaptive - Standard)')
plt.xlabel('Time Overhead (ms)')
plt.ylabel('Number of Images')
plt.axvline(mean_overhead, color='r', linestyle='--', label=f'Mean: {mean_overhead:.1f} ms')
plt.axvline(median_overhead, color='g', linestyle='-', label=f'Median: {median_overhead:.1f} ms')
plt.axvline(0, color='k', linestyle=':', label='No Difference (0 ms)')
plt.legend()
plt.grid(axis='y', alpha=0.5)

# --- Save Figure ---
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH)
print(f"Histogram saved to: {OUTPUT_PATH}")

# plt.show() # Uncomment if running interactively 