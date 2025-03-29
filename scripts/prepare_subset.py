#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path
import json

def create_subset(source_dir, target_dir, num_images=30, seed=42):
    """
    Create a subset of images from the source directory
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all image files from the source directory
    source_path = Path(source_dir)
    image_files = list(source_path.glob('*.jpg'))
    
    # Select a random subset of images
    if len(image_files) > num_images:
        selected_files = random.sample(image_files, num_images)
    else:
        selected_files = image_files
        print(f"Warning: Requested {num_images} images, but only {len(image_files)} available")
    
    # Copy the selected files to the target directory
    file_info = []
    for i, file in enumerate(selected_files):
        target_file = Path(target_dir) / file.name
        shutil.copy2(file, target_file)
        file_info.append({
            "id": i,
            "filename": file.name,
            "original_path": str(file),
            "subset_path": str(target_file)
        })
        print(f"Copied {file.name} ({i+1}/{len(selected_files)})")
    
    # Save metadata
    metadata = {
        "total_images": len(selected_files),
        "source_directory": str(source_dir),
        "target_directory": str(target_dir),
        "files": file_info
    }
    
    with open(Path(target_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Successfully created subset with {len(selected_files)} images in {target_dir}")
    print(f"Metadata saved to {os.path.join(target_dir, 'metadata.json')}")

def main():
    parser = argparse.ArgumentParser(description="Create a subset of images for experiment")
    parser.add_argument("--source", type=str, required=True, help="Source directory with images")
    parser.add_argument("--target", type=str, required=True, help="Target directory for subset")
    parser.add_argument("--num", type=int, default=30, help="Number of images in subset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    create_subset(args.source, args.target, args.num, args.seed)

if __name__ == "__main__":
    main() 