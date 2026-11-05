"""
High-precision coordinate-aware bitemporal crop extractor for Stage C.
Extracts corresponding patches from T1 and T2 given candidate bounding boxes.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image


def compute_padded_bbox(
    bbox: List[int],
    image_shape: Tuple[int, int],
    padding_fraction: float = 0.15,
    min_size: int = 16
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Computes a padded bounding box around the candidate region to provide
    contextual spatial information for structural verification.
    
    Args:
        bbox: [x0, y0, x1, y1] in half-open pixel coordinates.
        image_shape: (H, W) of the source image.
        padding_fraction: Relative padding to expand on all 4 sides.
        min_size: Minimum crop width and height in pixels.
        
    Returns:
        padded_bbox: [pad_x0, pad_y0, pad_x1, pad_y1] clamped to image borders.
        transform_meta: Metadata tracking padding and coordinate offsets.
    """
    H, W = image_shape[:2]
    x0, y0, x1, y1 = bbox
    
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    
    pad_w = max(int(w * padding_fraction), 2)
    pad_h = max(int(h * padding_fraction), 2)
    
    # Enforce minimum crop size
    if (w + 2 * pad_w) < min_size:
        extra_w = (min_size - w) // 2
        pad_w = max(pad_w, extra_w)
    if (h + 2 * pad_h) < min_size:
        extra_h = (min_size - h) // 2
        pad_h = max(pad_h, extra_h)
        
    pad_x0 = max(0, x0 - pad_w)
    pad_y0 = max(0, y0 - pad_h)
    pad_x1 = min(W, x1 + pad_w)
    pad_y1 = min(H, y1 + pad_h)
    
    transform_meta = {
        "original_bbox": [int(x) for x in bbox],
        "padded_bbox": [int(pad_x0), int(pad_y0), int(pad_x1), int(pad_y1)],
        "padding_applied": {"pad_x0": int(x0 - pad_x0), "pad_y0": int(y0 - pad_y0),
                            "pad_x1": int(pad_x1 - x1), "pad_y1": int(pad_y1 - y1)},
        "coordinate_frame": "original_T1_pixels"
    }
    
    return [pad_x0, pad_y0, pad_x1, pad_y1], transform_meta


def extract_bitemporal_crops(
    image_t1: np.ndarray,
    image_t2: np.ndarray,
    bbox: List[int],
    padding_fraction: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Extracts identical coordinate crops from paired T1 and T2 images.
    
    Args:
        image_t1: RGB array of shape (H, W, 3) at T1 (earlier).
        image_t2: RGB array of shape (H, W, 3) at T2 (later).
        bbox: [x0, y0, x1, y1] region coordinates.
        padding_fraction: Context window padding.
        
    Returns:
        crop_t1: Cropped uint8 RGB patch from T1.
        crop_t2: Cropped uint8 RGB patch from T2.
        meta: Spatial transform metadata.
    """
    assert image_t1.shape[:2] == image_t2.shape[:2], (
        f"Spatial dimension mismatch: T1={image_t1.shape} vs T2={image_t2.shape}"
    )
    
    padded_box, meta = compute_padded_bbox(bbox, image_t1.shape[:2], padding_fraction)
    x0, y0, x1, y1 = padded_box
    
    crop_t1 = image_t1[y0:y1, x0:x1].copy()
    crop_t2 = image_t2[y0:y1, x0:x1].copy()
    
    return crop_t1, crop_t2, meta
