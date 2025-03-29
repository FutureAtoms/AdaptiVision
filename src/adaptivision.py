#!/usr/bin/env python
"""
AdaptiVision: An adaptive context-aware object detection system
with scene complexity analysis and adaptive confidence thresholding.
"""

import os
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path

try:
    from ultralytics import YOLO
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

class AdaptiVision:
    """
    AdaptiVision: Scene-adaptive object detection with contextual reasoning.
    
    The model dynamically adjusts confidence thresholds based on scene complexity,
    object relationships, and class-specific detection challenges.
    """
    
    def __init__(
        self, 
        model_path: str = 'weights/model_n.pt',
        input_size: int = 640,
        device: str = 'auto',
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_adaptive_confidence: bool = True,
        context_aware: bool = True
    ):
        """
        Initialize AdaptiVision detector.
        
        Args:
            model_path: Path to model weights
            input_size: Input image size
            device: Device to run inference on ('auto', 'cpu', 'cuda', 'mps')
            conf_threshold: Base confidence threshold
            iou_threshold: IoU threshold for NMS
            enable_adaptive_confidence: Whether to use adaptive confidence thresholding
            context_aware: Whether to use context-aware reasoning
        """
        self.model_path = model_path
        self.input_size = input_size
        
        # If device is auto, choose appropriate device
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'has_mps') and torch.has_mps:
                device = 'mps'  # For Apple Silicon
            else:
                device = 'cpu'
                
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.enable_adaptive_confidence = enable_adaptive_confidence
        self.context_aware = context_aware
        
        # Knowledge base for object relationships
        self.object_relationships = {
            'person': ['chair', 'cup', 'bottle', 'cell phone', 'laptop', 'book', 'backpack'],
            'car': ['truck', 'bus', 'traffic light', 'stop sign', 'bicycle', 'motorcycle', 'person'],
            'chair': ['dining table', 'person', 'cup', 'bowl', 'book', 'laptop'],
            'dining table': ['chair', 'cup', 'bowl', 'fork', 'knife', 'spoon', 'pizza', 'person'],
            'dog': ['person', 'ball', 'cat', 'couch'],
            'bicycle': ['person', 'traffic light', 'stop sign'],
            'backpack': ['person'],
            'umbrella': ['person'],
            'handbag': ['person'],
            'tie': ['person'],
            'suitcase': ['person'],
            'laptop': ['person', 'chair', 'dining table'],
            'mouse': ['laptop', 'keyboard', 'person'],
            'keyboard': ['laptop', 'person', 'mouse'],
            'cell phone': ['person'],
            'oven': ['refrigerator', 'microwave', 'bowl'],
            'sink': ['toilet', 'cup', 'bowl', 'bottle'],
            'refrigerator': ['oven', 'microwave', 'bowl'],
            'book': ['chair', 'person', 'dining table']
        }
        
        # Class-specific confidence adjustments
        self.class_conf_adjustments = {
            # Small objects need lower thresholds, but more conservative than before
            'cell phone': -0.03,  # Was -0.05
            'mouse': -0.03,       # Was -0.05  
            'key': -0.03,         # Was -0.05
            'tie': -0.02,         # Was -0.05 - more conservative to reduce false positives
            'bottle': -0.02,      # Was -0.03
            'wine glass': -0.02,  # Was -0.03
            'cup': -0.02,         # Was -0.03
            'fork': -0.03,        # Was -0.05
            'knife': -0.03,       # Was -0.05
            'spoon': -0.03,       # Was -0.05
            # Hard-to-distinguish objects need lower thresholds
            'chair': -0.02,
            'dining table': -0.02,
            'sofa': -0.02,
            'bowl': -0.02,        # Was -0.03
            'laptop': -0.02,
            'remote': -0.03,      # Was -0.05
            'handbag': -0.02,     # Added to reduce false positives
            'bicycle': 0.0,       # Added to reduce false positives
            # Clear, large objects can have higher thresholds
            'person': 0.01,       # Reduced from 0.02 to improve person detection
            'car': 0.03,
            'bus': 0.03,
            'truck': 0.03,
            'airplane': 0.05,
            'train': 0.05,
            'boat': 0.02,
            'traffic light': -0.02,  # Changed from 0.0 to -0.02 to improve detection
            'fire hydrant': 0.01,    # Reduced from 0.02
            'stop sign': -0.03       # Changed from 0.05 to -0.03 to ensure detection
        }
        
        # Scene complexity factors weight (must sum to 1.0)
        self.complexity_factors = {
            'num_objects': 0.5,      # More objects → more complex
            'object_size_var': 0.25,  # High variance in size → more complex
            'object_density': 0.25    # High density → more complex
        }
        
        # Load model
        if not BACKEND_AVAILABLE:
            print("Warning: Ultralytics backend not available. Install with: pip install ultralytics")
            self.model = None
        else:
            try:
                self.model = YOLO(model_path)
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        
    def predict(
        self, 
        image_path: Union[str, np.ndarray],
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False,
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Run object detection on an image.
        
        Args:
            image_path: Path to image or image array
            conf_threshold: Optional override for confidence threshold
            iou_threshold: Optional override for IoU threshold
            classes: Optional list of classes to detect
            verbose: Whether to print detection details
            output_path: Optional path to save visualization
            
        Returns:
            List of dictionaries with detection results
        """
        if self.model is None:
            print("Error: Model not loaded")
            return []
            
        # Use provided thresholds or defaults
        conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_threshold = iou_threshold if iou_threshold is not None else self.iou_threshold
        
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                print(f"Error: Image path {image_path} does not exist")
                return []
            
            # Check file permissions
            if not os.access(image_path, os.R_OK):
                print(f"Error: No read permission for {image_path}")
                return []
                
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                print(f"Error: File {image_path} is empty (0 bytes)")
                return []
                
            # Print file information for debugging
            print(f"Loading image: {image_path} (Size: {file_size} bytes)")
            
            # Get original image for visualization
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Error: Could not read image {image_path}")
                # Try with PIL as a fallback
                try:
                    from PIL import Image
                    pil_img = np.array(Image.open(image_path))
                    if pil_img.shape[2] == 4:  # handle RGBA
                        pil_img = pil_img[:, :, :3]
                    # Convert from RGB to BGR for OpenCV compatibility
                    original_img = pil_img[:, :, ::-1].copy()
                    print(f"Successfully loaded image with PIL fallback, shape: {original_img.shape}")
                except Exception as e:
                    print(f"Failed to load with PIL fallback: {e}")
                    return []
        else:
            # Use provided image array
            original_img = image_path.copy()
        
        # Initial confidence threshold (lowered to get potential objects)
        initial_conf = min(0.05, conf_threshold * 0.5) if self.enable_adaptive_confidence else conf_threshold
        
        # Run detection
        results = self.model(
            image_path, 
            conf=initial_conf,
            iou=iou_threshold,
            classes=classes,
            verbose=False
        )
        
        # Process results
        all_results = []
        
        for i, result in enumerate(results):
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Get class names
            class_names = [result.names[int(cls_id)] for cls_id in class_ids]
            
            # Get image shape for visualization
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                img_shape = img.shape
            else:
                img_shape = image_path.shape
            
            if len(boxes) == 0:
                # No detections
                detection_result = {
                    'boxes': np.array([]),
                    'scores': np.array([]),
                    'labels': np.array([]),
                    'class_names': [],
                    'image_path': image_path if isinstance(image_path, str) else 'array',
                    'image_shape': img_shape,
                    'inference_time': time.time() - start_time
                }
                
                all_results.append(detection_result)
                continue
            
            # Process with adaptive confidence if enabled
            if self.enable_adaptive_confidence:
                # Calculate scene complexity
                scene_complexity = self._calculate_scene_complexity(boxes, class_names, img_shape)
                
                # Get adaptive threshold
                adaptive_threshold = self._get_adaptive_threshold(conf_threshold, scene_complexity)
                
                # Apply class-specific adjustments and context-aware reasoning
                adjusted_scores = scores.copy()
                final_boxes = []
                final_scores = []
                final_labels = []
                final_names = []
                
                for i, (box, score, label, name) in enumerate(zip(boxes, scores, class_ids, class_names)):
                    # Get class-specific adjustment
                    class_adj = self.class_conf_adjustments.get(name, 0.0)
                    
                    # Apply context-aware reasoning if enabled
                    context_boost = 0.0
                    if self.context_aware:
                        context_boost = self._apply_context_reasoning(name, class_names, i)
                    
                    # Calculate final threshold for this class
                    class_threshold = max(0.10, min(0.95, adaptive_threshold + class_adj))
                    
                    # Apply context boost to score
                    adjusted_score = min(1.0, score + context_boost)
                    
                    # Keep detection if score exceeds class-specific threshold
                    if adjusted_score >= class_threshold:
                        # Add to candidates
                        final_boxes.append(box)
                        final_scores.append(adjusted_score)
                        final_labels.append(label)
                        final_names.append(name)
                
                # Apply post-processing validation to filter out likely false positives
                filtered_boxes, filtered_scores, filtered_labels, filtered_names = self._post_process_detections(
                    np.array(final_boxes) if final_boxes else np.array([]),
                    np.array(final_scores) if final_scores else np.array([]),
                    np.array(final_labels) if final_labels else np.array([]),
                    final_names,
                    img_shape
                )
                
                # Create result dictionary
                detection_result = {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'labels': filtered_labels,
                    'class_names': filtered_names,
                    'image_path': image_path if isinstance(image_path, str) else 'array',
                    'image_shape': img_shape,
                    'inference_time': time.time() - start_time,
                    'scene_complexity': scene_complexity,
                    'adaptive_threshold': adaptive_threshold,
                    'enable_adaptive': self.enable_adaptive_confidence,
                    'context_aware': self.context_aware
                }
            else:
                # Filter by confidence without adaptive thresholding
                mask = scores >= conf_threshold
                filtered_boxes = boxes[mask]
                filtered_scores = scores[mask]
                filtered_labels = class_ids[mask]
                filtered_names = [class_names[i] for i, m in enumerate(mask) if m]
                
                # Create result dictionary
                detection_result = {
                    'boxes': filtered_boxes,
                    'scores': filtered_scores,
                    'labels': filtered_labels,
                    'class_names': filtered_names,
                    'image_path': image_path if isinstance(image_path, str) else 'array',
                    'image_shape': img_shape,
                    'inference_time': time.time() - start_time
                }
            
            all_results.append(detection_result)
        
        # Print results if verbose
        if verbose and all_results:
            self._print_detection_summary(all_results[0])
        
        # Save visualization if output path provided
        if output_path and all_results:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Visualize detections
            vis_img = self.visualize(
                image_path if isinstance(image_path, str) else original_img,
                all_results[0],
                output_path=output_path
            )
        
        return all_results
    
    def _calculate_scene_complexity(self, boxes: np.ndarray, class_names: List[str], image_shape: Tuple[int, int, int]) -> float:
        """Calculate scene complexity based on object count, size variance, and density."""
        if len(boxes) == 0:
            return 0.3  # Default moderate complexity
        
        # Number of objects (normalized with log scale)
        num_objects = len(boxes)
        norm_num_objects = min(1.0, np.log2(num_objects + 1) / 5.0)  # Log scale to handle large numbers
        
        # Object size variance
        box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        if len(box_areas) > 1:
            size_var = np.std(box_areas) / (np.mean(box_areas) + 1e-5)  # Normalized variance
            norm_size_var = min(1.0, size_var / 2.0)  # Cap at 1.0
        else:
            norm_size_var = 0.0
        
        # Object density (total box area / image area)
        img_area = image_shape[0] * image_shape[1]
        total_box_area = sum(box_areas)
        box_coverage = total_box_area / img_area
        norm_density = min(1.0, box_coverage * 3.0)  # Higher weight to density
        
        # Combine factors with weights
        complexity = (
            self.complexity_factors['num_objects'] * norm_num_objects +
            self.complexity_factors['object_size_var'] * norm_size_var +
            self.complexity_factors['object_density'] * norm_density
        )
        
        return complexity
    
    def _get_adaptive_threshold(self, base_threshold: float, scene_complexity: float) -> float:
        """
        Calculate adaptive threshold based on scene complexity.
        
        Higher complexity scenes receive lower thresholds to catch more objects.
        Lower complexity scenes receive higher thresholds to reduce false positives.
        """
        # Adjustment range - balance between false positives and missed detections
        min_adjust = -0.12  # Slightly more reduction than previous (-0.10) to catch important objects
        max_adjust = 0.05   # Maximum increase for simple scenes
        
        # Non-linear mapping for balanced adaptation
        if scene_complexity < 0.3:
            # Low complexity - nearly linear adjustment toward max_adjust
            factor = scene_complexity / 0.3
            adjustment = max_adjust * (1 - factor)
        elif scene_complexity > 0.7:
            # High complexity - more aggressive adjustment toward min_adjust
            factor = (scene_complexity - 0.7) / 0.3
            adjustment = (max_adjust - min_adjust) * (1 - factor) + min_adjust
        else:
            # Mid complexity - more balanced adjustment
            normalized = (scene_complexity - 0.3) / 0.4  # 0 to 1 in mid-range
            # Less conservative curve in the middle range
            mid_factor = 0.5 * (normalized ** 2) + 0.5 * normalized  # Equal weighting of linear and quadratic
            adjustment = max_adjust - mid_factor * (max_adjust - min_adjust)
        
        # Apply adjustment with bounds
        new_threshold = base_threshold + adjustment
        new_threshold = max(0.08, min(0.95, new_threshold))  # Reduce minimum from 0.10 to 0.08
        
        return new_threshold
    
    def _apply_context_reasoning(self, obj_name: str, all_class_names: List[str], obj_idx: int) -> float:
        """
        Apply context-aware reasoning to boost confidence.
        
        Objects that commonly co-occur with other detected objects receive a confidence boost.
        Certain objects require stronger contextual evidence to be considered valid.
        """
        # Important traffic objects should have a baseline boost to improve detection
        traffic_objects = {
            'stop sign': 0.03,
            'traffic light': 0.03,
            'bicycle': 0.02
        }
        
        # Apply base boost for important traffic objects
        base_boost = traffic_objects.get(obj_name, 0.0)
        
        if obj_name not in self.object_relationships:
            return base_boost
        
        # Get related objects for this class
        related_objects = self.object_relationships[obj_name]
        
        if not related_objects:
            return base_boost
            
        # Count how many related objects are in the scene
        related_count = sum(1 for name in all_class_names 
                           if name in related_objects and name != obj_name)
        
        # No related objects found - apply penalty to objects that are rarely seen alone
        if related_count == 0:
            # Define objects that rarely appear alone
            rarely_alone_objects = {
                'tie': -0.05,       # Ties usually appear with persons
                'handbag': -0.04,   # Handbags usually appear with persons
                'fork': -0.05,      # Forks usually appear with dining tables
                'knife': -0.05,     # Knives usually appear with dining tables
                'spoon': -0.05,     # Spoons usually appear with dining tables
                'cell phone': -0.03 # Cell phones usually appear with persons
            }
            
            if obj_name in rarely_alone_objects:
                return rarely_alone_objects[obj_name] + base_boost
            return base_boost
        
        # Different objects need different levels of contextual support
        # Some objects (like ties) need more contextual evidence than others
        context_requirements = {
            'tie': 0.5,         # Less strict requirement (was 0.6)
            'cell phone': 0.4,  # Less strict requirement (was 0.5)
            'handbag': 0.4,     # Less strict requirement (was 0.5)
            'fork': 0.5,
            'knife': 0.5,
            'spoon': 0.5
        }
        
        # Default requirement - 30% of related objects should be present for max boost
        requirement = context_requirements.get(obj_name, 0.3)
        
        # Scale boost based on how many related objects are present relative to requirement
        max_boost = 0.09  # Slightly higher maximum confidence boost (was 0.08)
        normalized_count = min(1.0, related_count / (len(related_objects) * requirement))
        boost = max_boost * normalized_count
        
        return boost + base_boost
        
    def visualize(
        self,
        image_path: Union[str, np.ndarray],
        detections: Dict,
        output_path: Optional[str] = None,
        line_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1
    ) -> np.ndarray:
        """
        Visualize detections on an image.
        
        Args:
            image_path: Path to image or image array
            detections: Detection results dictionary
            output_path: Optional path to save visualization
            line_thickness: Bounding box thickness
            font_scale: Text size
            font_thickness: Text thickness
            
        Returns:
            Annotated image array
        """
        # Load image if path provided
        if isinstance(image_path, str):
            print(f"Loading image for visualization: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path} for visualization")
                # Try with PIL as a fallback
                try:
                    from PIL import Image
                    pil_img = np.array(Image.open(image_path))
                    if pil_img.shape[2] == 4:  # handle RGBA
                        pil_img = pil_img[:, :, :3]
                    # Convert from RGB to BGR for OpenCV compatibility
                    image = pil_img[:, :, ::-1].copy()
                    print(f"Successfully loaded image with PIL fallback, shape: {image.shape}")
                except Exception as e:
                    print(f"Failed to load with PIL fallback: {e}")
                    return None
        else:
            image = image_path.copy()
        
        # Extract detection components
        boxes = detections.get('boxes', np.array([]))
        scores = detections.get('scores', np.array([]))
        labels = detections.get('labels', np.array([]))
        class_names = detections.get('class_names', [])
        
        # Draw bounding boxes and labels
        if len(boxes) > 0:
            image = self._draw_boxes(
                image, boxes, scores, labels, class_names,
                line_thickness, font_scale, font_thickness
            )
        
        # Add adaptive threshold info if available
        if 'adaptive_threshold' in detections and 'scene_complexity' in detections:
            # Add info about adaptive threshold
            complexity = detections['scene_complexity']
            adaptive_threshold = detections['adaptive_threshold']
            
            # Choose text color based on complexity (red for high, green for low)
            if complexity > 0.7:
                color = (0, 0, 255)  # Red for high complexity
            elif complexity > 0.4:
                color = (0, 165, 255)  # Orange for medium complexity
            else:
                color = (0, 255, 0)  # Green for low complexity
                
            # Add text at the top of the image
            text = f"Scene complexity: {complexity:.2f} | Adaptive threshold: {adaptive_threshold:.2f}"
            cv2.putText(
                image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2, cv2.LINE_AA
            )
        
        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, image)
            
        return image
    
    def _draw_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        line_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1
    ) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        # Generate colors for classes
        colors = self._generate_colors(max(labels) + 1 if len(labels) > 0 else 80)
        
        # Copy image to avoid modifying original
        result_img = image.copy()
        h, w = image.shape[:2]
        
        # Draw each box
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            # Get coordinates
            x1, y1, x2, y2 = box.astype(int)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Get color for class
            color = colors[int(label) % len(colors)]
            
            # Draw box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, line_thickness)
            
            # Create label text
            if i < len(class_names):
                class_name = class_names[i]
                label_text = f"{class_name} {score:.2f}"
            else:
                label_text = f"Class {label} {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                result_img, 
                (x1, y1 - text_height - baseline - 5), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_img,
                label_text,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
        
        return result_img
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        np.random.seed(42)  # For reproducibility
        
        # Generate vibrant colors
        colors = []
        for i in range(num_classes):
            # Use HSV color space for distinct colors
            hue = i / num_classes
            sat = 0.8 + np.random.rand() * 0.2  # High saturation for vibrant colors
            val = 0.8 + np.random.rand() * 0.2  # Bright colors
            
            # Convert to RGB
            h = hue * 360
            s = sat * 100
            v = val * 100
            
            # HSV to RGB conversion
            h_i = int(h / 60) % 6
            f = h / 60 - h_i
            p = v * (1 - s / 100)
            q = v * (1 - f * s / 100)
            t = v * (1 - (1 - f) * s / 100)
            
            if h_i == 0:
                r, g, b = v, t, p
            elif h_i == 1:
                r, g, b = q, v, p
            elif h_i == 2:
                r, g, b = p, v, t
            elif h_i == 3:
                r, g, b = p, q, v
            elif h_i == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
            
            # Scale to 0-255
            rgb = (int(r * 255 / 100), int(g * 255 / 100), int(b * 255 / 100))
            colors.append(rgb)
        
        return colors
    
    def _print_detection_summary(self, detections: Dict) -> None:
        """Print a summary of detection results."""
        boxes = detections.get('boxes', np.array([]))
        scores = detections.get('scores', np.array([]))
        class_names = detections.get('class_names', [])
        
        if len(boxes) == 0:
            print("No objects detected")
            return
        
        # Count objects by class
        class_counts = {}
        for name in class_names:
            if name in class_counts:
                class_counts[name] += 1
            else:
                class_counts[name] = 1
        
        # Print summary
        print("\n--- Detection Summary ---")
        print(f"Found {len(boxes)} objects:")
        
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count}")
        
        # Print adaptive info if available
        if 'scene_complexity' in detections and 'adaptive_threshold' in detections:
            print(f"\nScene complexity: {detections['scene_complexity']:.2f}")
            print(f"Adaptive threshold: {detections['adaptive_threshold']:.3f}")
            
            if 'enable_adaptive' in detections and detections['enable_adaptive']:
                threshold_diff = detections['adaptive_threshold'] - self.conf_threshold
                if threshold_diff < 0:
                    print(f"Threshold lowered by {abs(threshold_diff):.3f} due to scene complexity")
                else:
                    print(f"Threshold raised by {threshold_diff:.3f} due to scene simplicity")
        
        print(f"Inference time: {detections['inference_time']:.4f} seconds")
        print("-----------------------")

    def _post_process_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        img_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Apply post-processing validation to filter out likely false positives.
        
        Uses size validation, boundary checks, and class-specific constraints.
        """
        if len(boxes) == 0:
            return boxes, scores, labels, class_names
            
        # Initialize masks for valid detections
        valid_mask = np.ones(len(boxes), dtype=bool)
        h, w = img_shape[:2]
        img_area = h * w
        
        # Calculate areas and aspect ratios
        areas = np.array([(box[2] - box[0]) * (box[3] - box[1]) for box in boxes])
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        aspect_ratios = width / np.maximum(height, 1e-6)  # Avoid division by zero
        
        # Map from class name to index
        name_to_idx = {name: i for i, name in enumerate(class_names)}
        
        # Class-specific validations - adjusted to be less aggressive for important objects
        problematic_classes = {
            'tie': {'min_aspect': 0.15, 'max_aspect': 0.8, 'min_rel_area': 0.0008, 'min_score': 0.25},
            'cell phone': {'min_aspect': 0.4, 'max_aspect': 2.2, 'min_rel_area': 0.0004, 'min_score': 0.25},
            'handbag': {'min_aspect': 0.6, 'max_aspect': 1.7, 'min_rel_area': 0.0008, 'min_score': 0.22},
            'bicycle': {'min_aspect': 0.7, 'max_aspect': 2.7, 'min_rel_area': 0.004, 'min_score': 0.18}
        }
        
        # List of important classes that should be less filtered (safety-critical objects)
        important_classes = ['stop sign', 'traffic light', 'person', 'car', 'bus', 'truck']
        
        # Apply class-specific constraints
        for class_name, constraints in problematic_classes.items():
            if class_name in class_names:
                # Find all instances of this class
                class_indices = [i for i, name in enumerate(class_names) if name == class_name]
                
                for idx in class_indices:
                    aspect = aspect_ratios[idx]
                    rel_area = areas[idx] / img_area
                    
                    # Check if detection violates any constraints
                    if (aspect < constraints['min_aspect'] or 
                        aspect > constraints['max_aspect'] or
                        rel_area < constraints['min_rel_area'] or
                        scores[idx] < constraints['min_score']):
                        valid_mask[idx] = False
        
        # Filter out small detections with low scores (likely noise)
        rel_areas = areas / img_area
        for i, (area, score, name) in enumerate(zip(rel_areas, scores, class_names)):
            # Important classes have lower thresholds
            is_important = name in important_classes
            
            # Very small objects should have higher confidence, unless they're important classes
            if area < 0.001 and not is_important and score < 0.33:
                valid_mask[i] = False
            elif area < 0.001 and is_important and score < 0.20:  # More permissive for important classes
                valid_mask[i] = False
                
            # Penalize objects touching image boundaries (often partial detections)
            box = boxes[i]
            touches_boundary = (
                box[0] <= 1 or box[1] <= 1 or  # Left or top edge
                box[2] >= w-1 or box[3] >= h-1  # Right or bottom edge
            )
            
            # Less strict for important classes touching boundaries
            if touches_boundary and not is_important and score < 0.35:
                valid_mask[i] = False
            elif touches_boundary and is_important and score < 0.22:
                valid_mask[i] = False
        
        # Filter detections
        filtered_boxes = boxes[valid_mask]
        filtered_scores = scores[valid_mask]
        filtered_labels = labels[valid_mask]
        filtered_names = [class_names[i] for i, m in enumerate(valid_mask) if m]
        
        return filtered_boxes, filtered_scores, filtered_labels, filtered_names

import torch  # Required for device checks

def main():
    """Run a demo of AdaptiVision on a sample image."""
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="AdaptiVision: Adaptive Context-Aware Object Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="weights/model_n.pt", help="Path to model weights")
    parser.add_argument("--output", type=str, default="output/detected.jpg", help="Path to output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--adaptive", action="store_true", help="Enable adaptive confidence thresholding")
    parser.add_argument("--context", action="store_true", help="Enable context-aware reasoning")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on ('auto', 'cpu', 'cuda', 'mps')")
    parser.add_argument("--classes", type=int, nargs="+", help="Filter by class")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} does not exist")
        return
    
    # Check if model exists
    if not os.path.exists(args.weights):
        print(f"Error: Model {args.weights} does not exist")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Create detector
    detector = AdaptiVision(
        model_path=args.weights,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        enable_adaptive_confidence=args.adaptive,
        context_aware=args.context
    )
    
    # Run detection
    results = detector.predict(
        args.image,
        verbose=True,
        output_path=args.output,
        classes=args.classes
    )
    
    # Print summary if results
    if results:
        print(f"\nDetection results saved to {args.output}")

if __name__ == "__main__":
    main() 