#!/usr/bin/env python
"""
Command Line Interface for AdaptiVision

This script provides a unified command-line interface for the AdaptiVision project.
It includes subcommands for detection, visualization, comparison, and batch processing.
"""
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import AdaptiVision modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import AdaptiVision modules
from src.adaptivision import AdaptiVision
import src.create_visualizations as visualizations
import src.compare_methods as compare

def detect_command(args):
    """Detect objects in an image"""
    # Initialize detector
    detector = AdaptiVision(
        model_path=args.weights,
        device=args.device,
        conf_threshold=args.conf_thres,
        iou_threshold=args.iou_thres,
        enable_adaptive_confidence=not args.disable_adaptive,
        context_aware=not args.disable_context
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run detection
    print(f"Running detection on {args.image}")
    results = detector.predict(args.image)
    
    # Process results
    if results and len(results) > 0:
        result = results[0]
        print(f"Found {len(result['boxes'])} objects")
        
        # Print adaptive threshold info if enabled
        if not args.disable_adaptive and 'adaptive_threshold' in result:
            print(f"Scene complexity: {result['scene_complexity']:.3f}")
            print(f"Adaptive threshold: {result['adaptive_threshold']:.3f}")
    else:
        print("No objects detected")
    
    # Save detection visualization
    detector.visualize(args.image, results[0], args.output)
    print(f"Results saved to {args.output}")

def visualize_command(args):
    """Create detailed visualizations of adaptive thresholding"""
    # Modify sys.argv to match what the visualization script expects
    sys.argv = [
        'create_visualizations.py',
        '--image', args.image,
        '--weights', args.weights,
        '--output-dir', args.output_dir,
        '--device', args.device
    ]
    
    # Run the visualization module's main function
    visualizations.main()

def compare_command(args):
    """Compare standard vs. adaptive detection results"""
    # Modify sys.argv to match what the comparison script expects
    sys.argv = [
        'compare_methods.py',
        '--image', args.image,
        '--weights', args.weights,
        '--conf', str(args.conf_thres),
        '--iou', str(args.iou_thres),
        '--output-dir', args.output_dir,
        '--device', args.device
    ]
    
    # Run the comparison module's main function
    compare.main()

def batch_command(args):
    """Run batch processing on multiple images"""
    # Import the batch processing module here to avoid circular imports
    sys.path.insert(0, os.path.join(parent_dir, 'examples'))
    import batch_processing
    
    # Modify sys.argv to match what the batch processing script expects
    sys.argv = [
        'batch_processing.py',
        '--input-dir', args.input_dir,
        '--output-dir', args.output_dir,
        '--weights', args.weights,
        '--conf-thres', str(args.conf_thres),
        '--iou-thres', str(args.iou_thres),
        '--device', args.device,
        '--workers', str(args.workers)
    ]
    
    if args.disable_adaptive:
        sys.argv.append('--disable-adaptive')
    if args.disable_context:
        sys.argv.append('--disable-context')
    if args.save_json:
        sys.argv.append('--save-json')
    
    # Run the batch processing module's main function
    batch_processing.main()

def main():
    """Main CLI entry point"""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='AdaptiVision: Adaptive Context-Aware Object Detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create the parser for the "detect" command
    detect_parser = subparsers.add_parser('detect', help='Detect objects in an image')
    detect_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    detect_parser.add_argument('--weights', type=str, default='weights/model_n.pt', help='Path to model weights')
    detect_parser.add_argument('--output', type=str, default='results/detection.jpg', help='Path to output image')
    detect_parser.add_argument('--conf-thres', type=float, default=0.25, help='Base confidence threshold')
    detect_parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    detect_parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto, cpu, cuda, mps)')
    detect_parser.add_argument('--disable-adaptive', action='store_true', help='Disable adaptive confidence')
    detect_parser.add_argument('--disable-context', action='store_true', help='Disable context-aware reasoning')
    
    # Create the parser for the "visualize" command
    visualize_parser = subparsers.add_parser('visualize', help='Create visualizations of adaptive thresholding')
    visualize_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    visualize_parser.add_argument('--weights', type=str, default='weights/model_n.pt', help='Path to model weights')
    visualize_parser.add_argument('--output-dir', type=str, default='results/visualizations', help='Directory for output images')
    visualize_parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto, cpu, cuda, mps)')
    
    # Create the parser for the "compare" command
    compare_parser = subparsers.add_parser('compare', help='Compare standard vs. adaptive detection')
    compare_parser.add_argument('--image', type=str, required=True, help='Path to input image')
    compare_parser.add_argument('--weights', type=str, default='weights/model_n.pt', help='Path to model weights')
    compare_parser.add_argument('--conf-thres', type=float, default=0.25, help='Base confidence threshold')
    compare_parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    compare_parser.add_argument('--output-dir', type=str, default='results/comparison', help='Directory for output images')
    compare_parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto, cpu, cuda, mps)')
    
    # Create the parser for the "batch" command
    batch_parser = subparsers.add_parser('batch', help='Process multiple images in batch mode')
    batch_parser.add_argument('--input-dir', type=str, required=True, help='Directory containing input images')
    batch_parser.add_argument('--output-dir', type=str, default='results/batch', help='Directory for output images')
    batch_parser.add_argument('--weights', type=str, default='weights/model_n.pt', help='Path to model weights')
    batch_parser.add_argument('--conf-thres', type=float, default=0.25, help='Base confidence threshold')
    batch_parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold for NMS')
    batch_parser.add_argument('--device', type=str, default='auto', help='Device to run on (auto, cpu, cuda, mps)')
    batch_parser.add_argument('--disable-adaptive', action='store_true', help='Disable adaptive confidence')
    batch_parser.add_argument('--disable-context', action='store_true', help='Disable context-aware reasoning')
    batch_parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers (0 for sequential)')
    batch_parser.add_argument('--save-json', action='store_true', help='Save detection results as JSON')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command is specified
    if not args.command:
        parser.print_help()
        return
    
    # Execute the appropriate command
    if args.command == 'detect':
        detect_command(args)
    elif args.command == 'visualize':
        visualize_command(args)
    elif args.command == 'compare':
        compare_command(args)
    elif args.command == 'batch':
        batch_command(args)

if __name__ == '__main__':
    main() 