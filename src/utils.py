"""
Utility functions for AdaptiVision
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Device selection helper
def select_device(device=''):
    """
    Select the best available device for running inference.
    
    Args:
        device (str): Preferred device ('auto', 'cpu', 'cuda', 'mps')
        
    Returns:
        torch.device: Selected device for computation
    """
    if device.lower() == 'auto':
        device = ''  # Default to auto-detect
    
    # Check CUDA
    cuda = device.lower() == 'cuda' or (device == '' and torch.cuda.is_available())
    if cuda:
        return torch.device('cuda')
    
    # Check MPS (Apple Silicon)
    mps = device.lower() == 'mps' or (device == '' and getattr(torch, 'has_mps', False))
    if mps:
        return torch.device('mps')
    
    # Default to CPU
    return torch.device('cpu')

# Image loading and preprocessing
def load_image(image_path, input_size=640):
    """
    Load and preprocess an image for inference
    
    Args:
        image_path (str): Path to the image file
        input_size (int): Target size for model input
        
    Returns:
        tuple: (original_image, preprocessed_image, (original_height, original_width))
    """
    # Read image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path  # Assume it's already a numpy array
        
    if img is None:
        raise ValueError(f"Cannot load image from {image_path}")
        
    # Store original dimensions
    original_shape = img.shape[:2]  # (height, width)
    
    # Resize and pad image to maintain aspect ratio
    padded_img, ratio, padding = resize_and_pad(img, new_shape=input_size)
    
    # Convert to float and normalize
    padded_img = padded_img.transpose((2, 0, 1))  # HWC to CHW
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    padded_img /= 255.0  # Normalize to [0, 1]
    
    # Create batch dimension
    padded_img = torch.from_numpy(padded_img).unsqueeze(0)
    
    return img, padded_img, original_shape, ratio, padding

def resize_and_pad(img, new_shape=640, color=(114, 114, 114)):
    """
    Resize and pad image while maintaining aspect ratio
    
    Args:
        img (numpy.ndarray): Image to resize
        new_shape (int): Target size
        color (tuple): Padding color
        
    Returns:
        tuple: (resized_padded_image, ratio, (dw, dh)) where
               ratio is the scale factor and (dw, dh) are padding dimensions
    """
    shape = img.shape[:2]  # Current shape (height, width)
    
    # Calculate ratio (new / old)
    r = min(new_shape / shape[0], new_shape / shape[1])
    
    # Calculate new unpadded dimensions
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    # Calculate padding
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw, dh = dw / 2, dh / 2  # Divide padding into equal parts
    
    # Resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)

def adjust_boxes_to_original(boxes, ratio, padding, original_shape):
    """
    Adjust bounding boxes back to original image dimensions
    
    Args:
        boxes (numpy.ndarray): Bounding boxes from model prediction
        ratio (float): Scale ratio used for resizing
        padding (tuple): Padding added (dw, dh)
        original_shape (tuple): Original image shape (height, width)
        
    Returns:
        numpy.ndarray: Adjusted bounding boxes
    """
    # If no boxes, return empty array
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    
    # Extract padding
    dw, dh = padding
    
    # Adjust to remove padding and rescale to original dimensions
    adjusted_boxes = boxes.copy()
    
    # Remove padding
    adjusted_boxes[:, [0, 2]] -= dw  # x coordinates
    adjusted_boxes[:, [1, 3]] -= dh  # y coordinates
    
    # Rescale to original dimensions
    adjusted_boxes /= ratio
    
    # Clip to image boundaries
    adjusted_boxes[:, [0, 2]] = np.clip(adjusted_boxes[:, [0, 2]], 0, original_shape[1])
    adjusted_boxes[:, [1, 3]] = np.clip(adjusted_boxes[:, [1, 3]], 0, original_shape[0])
    
    return adjusted_boxes

def draw_detections(image, boxes, scores, class_ids, class_names, adaptive_threshold=None, base_threshold=0.25):
    """
    Draw detection boxes on image with class names and confidence scores
    
    Args:
        image (numpy.ndarray): Original image to draw on
        boxes (list): Bounding boxes coordinates [x1, y1, x2, y2]
        scores (list): Confidence scores for each box
        class_ids (list): Class IDs for each box
        class_names (list): Class names for each box
        adaptive_threshold (float, optional): Adaptive threshold value for color-coding
        base_threshold (float, optional): Base threshold value for comparison
        
    Returns:
        numpy.ndarray: Image with drawn detections
    """
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    
    # Generate random colors for each class (for consistent coloring)
    np.random.seed(42)  # For reproducibility
    colors = {i: tuple(map(int, np.random.randint(0, 255, 3))) for i in range(len(COCO_CLASSES))}
    
    # Create thin border around the image
    cv2.rectangle(img_copy, (0, 0), (w-1, h-1), (200, 200, 200), 1)
    
    # Add adaptive threshold info if provided
    if adaptive_threshold is not None:
        # Determine color based on threshold adjustment
        if adaptive_threshold < base_threshold:
            # Lower threshold (more lenient)
            threshold_color = (0, 0, 255)  # Red
            threshold_text = f"Adaptive: {adaptive_threshold:.3f} (↓{base_threshold-adaptive_threshold:.3f})"
        elif adaptive_threshold > base_threshold:
            # Higher threshold (more strict)
            threshold_color = (0, 255, 0)  # Green
            threshold_text = f"Adaptive: {adaptive_threshold:.3f} (↑{adaptive_threshold-base_threshold:.3f})"
        else:
            # Unchanged
            threshold_color = (200, 200, 200)  # Gray
            threshold_text = f"Threshold: {adaptive_threshold:.3f}"
            
        # Add threshold text
        cv2.putText(img_copy, threshold_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, threshold_color, 2)
    
    # Draw each detection
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        
        # Get class details
        class_name = class_names[i] if i < len(class_names) else f"Class {class_id}"
        color = colors.get(class_id, (255, 0, 0))  # Default red if class_id not in colors
        
        # Different border style based on confidence
        if adaptive_threshold is not None:
            if score < adaptive_threshold:
                # Below threshold - dashed line (would be filtered out)
                draw_dashed_rectangle(img_copy, (x1, y1), (x2, y2), color, 1)
                continue
            elif score < base_threshold:
                # Below base but above adaptive - special highlight
                color = (0, 165, 255)  # Orange - rescued by adaptive threshold
            
        # Draw box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        label = f"{class_name}: {score:.2f}"
        
        # Calculate label background size
        (label_width, label_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img_copy, 
            (x1, y1 - label_height - 5), 
            (x1 + label_width + 5, y1), 
            color, 
            -1
        )
        
        # Draw label text
        cv2.putText(
            img_copy, 
            label, 
            (x1 + 3, y1 - 4), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    # Add detection count
    label_count = f"Objects: {len([s for s in scores if adaptive_threshold is None or s >= adaptive_threshold])}"
    cv2.putText(img_copy, label_count, (w - 120, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return img_copy

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed rectangle on an image"""
    # Draw top line
    draw_dashed_line(img, (pt1[0], pt1[1]), (pt2[0], pt1[1]), color, thickness, dash_length)
    # Draw right line
    draw_dashed_line(img, (pt2[0], pt1[1]), (pt2[0], pt2[1]), color, thickness, dash_length)
    # Draw bottom line
    draw_dashed_line(img, (pt2[0], pt2[1]), (pt1[0], pt2[1]), color, thickness, dash_length)
    # Draw left line
    draw_dashed_line(img, (pt1[0], pt2[1]), (pt1[0], pt1[1]), color, thickness, dash_length)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    """Draw a dashed line on an image"""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    if dashes == 0:
        cv2.line(img, pt1, pt2, color, thickness)
        return
        
    for i in range(dashes):
        start_ratio = i / dashes
        end_ratio = (i + 0.5) / dashes
        
        start_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * start_ratio), 
                    int(pt1[1] + (pt2[1] - pt1[1]) * start_ratio))
        end_pt = (int(pt1[0] + (pt2[0] - pt1[0]) * end_ratio), 
                  int(pt1[1] + (pt2[1] - pt1[1]) * end_ratio))
        
        cv2.line(img, start_pt, end_pt, color, thickness)

def save_output_image(image, output_path, filename=None, create_dirs=True):
    """
    Save detection image to output path
    
    Args:
        image (numpy.ndarray): Image to save
        output_path (str): Output directory or full file path
        filename (str, optional): Filename to use if output_path is a directory
        create_dirs (bool): Whether to create directories if they don't exist
        
    Returns:
        str: Path to saved image
    """
    # Handle output path
    if os.path.isdir(output_path) or output_path.endswith('/'):
        if create_dirs and not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        
        if filename is None:
            # Generate a timestamp filename
            import time
            filename = f"detection_{int(time.time())}.jpg"
            
        output_file = os.path.join(output_path, filename)
    else:
        output_file = output_path
        # Create parent directories if needed
        if create_dirs:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_file, image)
    return output_file

def overlay_text(image, text, position, font_scale=0.7, color=(0, 0, 0), thickness=2):
    """
    Add text overlay to image with a semi-transparent background
    
    Args:
        image (numpy.ndarray): Image to add text to
        text (str): Text to display
        position (tuple): (x, y) position for text
        font_scale (float): Font scale
        color (tuple): Text color
        thickness (int): Line thickness
        
    Returns:
        numpy.ndarray: Image with text overlay
    """
    img_copy = image.copy()
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    
    # Create semi-transparent background
    padding = 5
    background = np.zeros((text_height + 2*padding, text_width + 2*padding, 3), dtype=np.uint8)
    background.fill(255)  # White background
    
    # Create alpha mask
    alpha = 0.7
    x, y = position
    
    # Ensure background stays within image boundaries
    y_end = min(y + text_height + 2*padding, img_copy.shape[0])
    x_end = min(x + text_width + 2*padding, img_copy.shape[1])
    bg_height = y_end - y
    bg_width = x_end - x
    
    # Adjust background size if needed
    if bg_height < background.shape[0] or bg_width < background.shape[1]:
        background = background[:bg_height, :bg_width]
    
    # Apply semi-transparent background
    img_copy[y:y_end, x:x_end] = cv2.addWeighted(
        img_copy[y:y_end, x:x_end], 1 - alpha, background, alpha, 0
    )
    
    # Add text
    cv2.putText(
        img_copy, text, (x + padding, y + text_height + padding - baseline),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness
    )
    
    return img_copy

def calculate_average_precision(predictions, ground_truth, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) for object detection
    
    Args:
        predictions (list): List of predicted boxes with format [x1, y1, x2, y2, confidence, class_id]
        ground_truth (list): List of ground truth boxes with format [x1, y1, x2, y2, class_id]
        iou_threshold (float): IoU threshold for considering a detection as correct
        
    Returns:
        float: Average Precision value
    """
    # Group predictions by class
    predictions_by_class = {}
    for pred in predictions:
        class_id = int(pred[5])
        if class_id not in predictions_by_class:
            predictions_by_class[class_id] = []
        predictions_by_class[class_id].append(pred)
    
    # Group ground truth by class
    gt_by_class = {}
    for gt in ground_truth:
        class_id = int(gt[4])
        if class_id not in gt_by_class:
            gt_by_class[class_id] = []
        gt_by_class[class_id].append(gt)
    
    # Calculate AP for each class
    average_precisions = []
    
    for class_id in gt_by_class:
        if class_id not in predictions_by_class:
            # No predictions for this class
            if len(gt_by_class[class_id]) > 0:
                average_precisions.append(0)
            continue
        
        # Get predictions and ground truth for this class
        class_preds = sorted(predictions_by_class[class_id], key=lambda x: x[4], reverse=True)
        class_gt = gt_by_class[class_id]
        
        # Mark each ground truth as used or not
        gt_used = [False] * len(class_gt)
        
        # Compute precision and recall
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        for i, pred in enumerate(class_preds):
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            pred_box = pred[:4]
            
            for j, gt in enumerate(class_gt):
                if gt_used[j]:
                    continue
                    
                gt_box = gt[:4]
                iou = calculate_iou(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            # Check if detection is correct
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp[i] = 1
                gt_used[best_gt_idx] = True
            else:
                fp[i] = 1
        
        # Compute cumulative values
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # Compute precision and recall
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        recall = cumsum_tp / len(class_gt) if len(class_gt) > 0 else np.zeros_like(cumsum_tp)
        
        # Compute average precision using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        average_precisions.append(ap)
    
    # Return mean AP across all classes
    return np.mean(average_precisions) if len(average_precisions) > 0 else 0

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1 (list or numpy.ndarray): First box coordinates [x1, y1, x2, y2]
        box2 (list or numpy.ndarray): Second box coordinates [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou 