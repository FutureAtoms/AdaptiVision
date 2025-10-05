#!/usr/bin/env python3
"""
Download YOLOv8 model weights automatically.

This script downloads the YOLOv8 nano model weights if they don't already exist.
It's cross-platform and works on Windows, macOS, and Linux.
"""

import os
import sys
import urllib.request
from pathlib import Path

# Model configurations
MODELS = {
    'yolov8n': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'filename': 'model_n.pt',
        'size_mb': 6.2,
        'description': 'YOLOv8 Nano - Fast, lightweight model (recommended)'
    },
    'yolov8s': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'filename': 'model_s.pt',
        'size_mb': 22,
        'description': 'YOLOv8 Small - Better accuracy, slower'
    },
    'yolov8m': {
        'url': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
        'filename': 'model_m.pt',
        'size_mb': 52,
        'description': 'YOLOv8 Medium - High accuracy, requires more memory'
    }
}

def download_with_progress(url, output_path):
    """Download a file with progress bar."""
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f'\rDownloading: {percent}% ')
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, output_path, progress_hook)
        sys.stdout.write('\n')
        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False

def download_model(model_name='yolov8n', force=False):
    """Download specified model weights."""
    if model_name not in MODELS:
        print(f"Error: Unknown model '{model_name}'")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False

    model_info = MODELS[model_name]
    weights_dir = Path(__file__).parent.parent / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

    output_path = weights_dir / model_info['filename']

    # Check if already exists
    if output_path.exists() and not force:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Model weights already exist: {output_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        print("Use --force to re-download")
        return True

    print("=" * 60)
    print("AdaptiVision - Model Weights Download")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Description: {model_info['description']}")
    print(f"Expected size: ~{model_info['size_mb']} MB")
    print(f"Output: {output_path}")
    print(f"URL: {model_info['url']}")
    print("=" * 60)

    # Download
    print("Starting download...")
    success = download_with_progress(model_info['url'], output_path)

    if success:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Download complete!")
        print(f"  Location: {output_path}")
        print(f"  Size: {file_size_mb:.1f} MB")
        print()
        print("You can now run AdaptiVision:")
        print(f"  python src/cli.py detect --image samples/bus.jpg --weights {output_path}")
        return True
    else:
        print("✗ Download failed!")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        return False

def main():
    """Main function with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download YOLOv8 model weights for AdaptiVision',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download default model (nano)
  python scripts/download_weights.py

  # Download small model
  python scripts/download_weights.py --model yolov8s

  # Force re-download
  python scripts/download_weights.py --force

Available models:
  yolov8n - Nano (6.2 MB) - Fastest, recommended
  yolov8s - Small (22 MB) - Better accuracy
  yolov8m - Medium (52 MB) - Best accuracy
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n',
        choices=list(MODELS.keys()),
        help='Model to download (default: yolov8n)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if file exists'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models and exit'
    )

    args = parser.parse_args()

    # List models
    if args.list:
        print("Available models:")
        for name, info in MODELS.items():
            print(f"\n{name}:")
            print(f"  {info['description']}")
            print(f"  Size: ~{info['size_mb']} MB")
            print(f"  Filename: {info['filename']}")
        return 0

    # Download model
    success = download_model(args.model, args.force)
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
