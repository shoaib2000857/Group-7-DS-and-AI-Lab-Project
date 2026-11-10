"""
Visual Evidence Inspector for Stage C (v3 — Production Grounded).
Performs physical state change verification on cropped bitemporal satellite patches.
Analyzes structural edge gradients, radiometric contrast shifts, registration jitter,
and semantic claim consistency.

Major Hardening Fixes:
  - FIX-1: Radiometric Mean-Shift Normalization in LAB space to eliminate seasonal & sun-angle shifts.
  - FIX-2: Spatial Block Anomaly Analysis (Peak-to-Mean Ratio) to detect localized structures
           amidst high-texture natural satellite terrain (resolves Texas grassland baseline).
  - FIX-3: Orthogonal Edge Orientation Analysis (0°/90° rectilinear concentration).
  - FIX-4: Semantic Claim Gating — Verifies whether the observed evidence supports the SPECIFIC
           claimed object & action (e.g., prevents seasonal grass shifts from falsely verifying building claims).
  - FIX-5: Crop Size Validation (crops < 24x24 flagged as indeterminate to prevent small-patch noise).
  - FIX-6: Dynamic Visibility Assessment (detects underexposure, saturation, low contrast).
"""

from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np


class VisualEvidenceInspector:
    """
    Independent Visual Witness verifying whether physical changes claimed
    by captioners are supported by structural evidence in satellite image crops.
    """

    def __init__(
        self,
        edge_threshold: float = 35.0,
        structural_density_thresh: float = 0.04,
        jitter_tolerance_px: float = 2.5,
        min_crop_size: int = 24,
        confidence_thresh: float = 0.50
    ):
        self.edge_threshold = edge_threshold
        self.structural_density_thresh = structural_density_thresh
        self.jitter_tolerance_px = jitter_tolerance_px
        self.min_crop_size = min_crop_size
        self.confidence_thresh = confidence_thresh

    # ------------------------------------------------------------------ #
    #  Image Quality & Radiometric Normalization                          #
    # ------------------------------------------------------------------ #

    def assess_crop_visibility(self, crop_rgb: np.ndarray) -> str:
        """Dynamically evaluates crop image quality."""
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        std_val = float(np.std(gray))
        mean_val = float(np.mean(gray))

        if std_val < 8.0:
            return "low_contrast"
        elif mean_val < 25.0:
            return "underexposed"
        elif mean_val > 235.0:
            return "saturated"
        return "clear"

    def normalize_illumination_meanshift(
        self,
        crop_t1: np.ndarray,
        crop_t2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Radiometric Mean-Shift Normalization in LAB color space.
        Aligns T2 luminance to T1 mean luminance to eliminate seasonal and
        solar elevation lighting differences without compressing genuine object contrast.
        
        Returns:
            norm_t1: Reference T1 (unchanged).
            norm_t2: Radiometrically normalized T2.
            mean_lum_shift: Raw T2 - T1 luminance offset before correction.
        """
        lab1 = cv2.cvtColor(crop_t1, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab2 = cv2.cvtColor(crop_t2, cv2.COLOR_RGB2LAB).astype(np.float32)

        m1 = float(np.mean(lab1[:, :, 0]))
        m2 = float(np.mean(lab2[:, :, 0]))
        mean_lum_shift = float(m2 - m1)

        # Shift T2 luminance channel to match T1 baseline
        lab2[:, :, 0] = np.clip(lab2[:, :, 0] - mean_lum_shift, 0.0, 255.0)

        norm_t2 = cv2.cvtColor(lab2.astype(np.uint8), cv2.COLOR_LAB2RGB)
        return crop_t1.copy(), norm_t2, mean_lum_shift

    # ------------------------------------------------------------------ #
    #  Edge & Orientation Analysis                                       #
    # ------------------------------------------------------------------ #

    def compute_edge_density(self, image_rgb: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Computes Sobel edge density and orthogonal orientation ratio.
        Returns:
            density: Normalized edge density in [0, 1].
            ortho_ratio: Fraction of strong edges aligned along 0°/90° (±15°).
            edge_mask: Binary mask of strong edges.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)

        edge_mask = (mag > self.edge_threshold).astype(np.uint8)
        density = float(np.mean(edge_mask))

        total_strong = int(np.sum(edge_mask))
        if total_strong < 5:
            ortho_ratio = 0.333  # Default random isotropic baseline
        else:
            angles = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))  # [0, 90]
            ortho_mask = ((angles < 15.0) | (angles > 75.0)) & (edge_mask > 0)
            ortho_ratio = float(np.sum(ortho_mask) / max(total_strong, 1))

        return density, ortho_ratio, edge_mask

    def detect_subpixel_jitter(
        self,
        crop_t1: np.ndarray,
        crop_t2: np.ndarray
    ) -> Tuple[bool, float]:
        """Uses phase correlation on luminance to detect sensor translation jitter."""
        g1 = cv2.cvtColor(crop_t1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(crop_t2, cv2.COLOR_RGB2GRAY).astype(np.float32)

        h, w = g1.shape
        if h < 8 or w < 8:
            return False, 0.0

        win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(g1, g2, win)
        shift_dist = float(np.sqrt(shift[0]**2 + shift[1]**2))

        is_jitter = (0.5 < shift_dist <= self.jitter_tolerance_px) and (response > 0.40)
        return is_jitter, shift_dist

    # ------------------------------------------------------------------ #
    #  Localized Spatial Anomaly & SSIM Analysis                         #
    # ------------------------------------------------------------------ #

    def compute_localized_block_metrics(
        self,
        crop_t1: np.ndarray,
        norm_t2: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Divides the normalized difference map into spatial blocks to separate
        compact localized physical structures from diffuse seasonal lighting shifts.
        
        Returns:
            mean_block_diff: Average difference across all blocks.
            max_block_diff: Intensity of the highest changing block.
            peak_to_mean_ratio: max_block_diff / mean_block_diff.
        """
        g1 = cv2.cvtColor(crop_t1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(norm_t2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        diff = np.abs(g1 - g2)

        h, w = diff.shape
        block_size = min(16, min(h, w))
        n_by = max(1, h // block_size)
        n_bx = max(1, w // block_size)

        block_diffs = []
        for by in range(n_by):
            for bx in range(n_bx):
                b = diff[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
                block_diffs.append(float(np.mean(b)))

        block_diffs = np.array(block_diffs, dtype=np.float32)
        mean_b = float(np.mean(block_diffs))
        max_b = float(np.max(block_diffs)) if len(block_diffs) > 0 else 0.0
        peak_to_mean = max_b / (mean_b + 1e-5)

        return mean_b, max_b, peak_to_mean

    def compute_ssim_score(
        self,
        crop_t1: np.ndarray,
        crop_t2: np.ndarray,
        win_size: int = 7
    ) -> float:
        """Computes structural similarity index (SSIM) between crops."""
        g1 = cv2.cvtColor(crop_t1, cv2.COLOR_RGB2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(crop_t2, cv2.COLOR_RGB2GRAY).astype(np.float64)

        h, w = g1.shape
        if h < win_size or w < win_size:
            return 1.0 if np.allclose(g1, g2) else 0.5

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        kernel = cv2.getGaussianKernel(win_size, 1.5)
        window = kernel @ kernel.T

        mu1 = cv2.filter2D(g1, -1, window)
        mu2 = cv2.filter2D(g2, -1, window)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu12 = mu1 * mu2

        sigma1_sq = cv2.filter2D(g1 ** 2, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(g2 ** 2, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(g1 * g2, -1, window) - mu12

        ssim_map = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.clip(np.mean(ssim_map), -1.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Primary Inspection Workflow                                        #
    # ------------------------------------------------------------------ #

    def inspect_crop_pair(
        self,
        crop_t1: np.ndarray,
        crop_t2: np.ndarray,
        claimed_object: str = "building",
        claimed_action: str = "constructed"
    ) -> Dict[str, Any]:
        """
        Conducts objective verification of claimed physical changes.
        """
        h, w = crop_t1.shape[:2]
        quality_flags = []
        visibility_flags = []

        # Check crop size validity
        if h < self.min_crop_size or w < self.min_crop_size:
            return {
                "visible_change": None,
                "confidence": 0.0,
                "state_transition": "indeterminate_small_crop",
                "observations": f"Crop dimensions ({w}x{h} px) are below reliable threshold ({self.min_crop_size} px). Evidence indeterminate.",
                "visibility_flags": ["small_crop"],
                "quality_flags": ["crop_too_small"],
                "structural_edge_change": 0.0,
                "mean_luminance_diff": 0.0,
                "crop_mean_diff": 0.0,
                "ssim_score": 1.0
            }

        # 1. Image Visibility Assessment
        vis1 = self.assess_crop_visibility(crop_t1)
        vis2 = self.assess_crop_visibility(crop_t2)
        visibility_flags.extend([f"t1_{vis1}", f"t2_{vis2}"])

        # 2. Illumination Mean-Shift Normalization
        norm_t1, norm_t2, lum_shift = self.normalize_illumination_meanshift(crop_t1, crop_t2)
        if abs(lum_shift) > 20.0:
            quality_flags.append(f"illumination_corrected_{lum_shift:+.1f}")

        # 3. Edge Density & Orientation Analysis (on radiometrically normalized crops)
        e1_density, ortho1, _ = self.compute_edge_density(norm_t1)
        e2_density, ortho2, _ = self.compute_edge_density(norm_t2)
        delta_edge = float(e2_density - e1_density)
        delta_ortho = float(ortho2 - ortho1)

        # 4. Subpixel Jitter Screening
        is_jitter, shift_px = self.detect_subpixel_jitter(crop_t1, crop_t2)
        if is_jitter:
            quality_flags.append(f"subpixel_jitter_{shift_px:.1f}px")

        # 5. Localized Block Anomaly Metrics
        mean_block_diff, max_block_diff, peak_ratio = self.compute_localized_block_metrics(norm_t1, norm_t2)

        # 6. SSIM Metric
        ssim_score = self.compute_ssim_score(norm_t1, norm_t2)

        # 7. Overall Mean Difference on normalized images
        abs_diff_norm = cv2.absdiff(norm_t1, norm_t2)
        crop_mean_diff = float(np.mean(abs_diff_norm))

        # 8. Shadow / Lighting Hazard Guard
        if abs(lum_shift) > 30.0 and abs(delta_edge) < 0.03 and peak_ratio < 1.80:
            quality_flags.append("shadow_hazard")

        if not quality_flags:
            quality_flags.append("registered_good_contrast")

        # -------------------------------------------------------------- #
        #  Semantic Claim Gating & State Transition Logic                 #
        # -------------------------------------------------------------- #
        obj_lower = claimed_object.lower() if claimed_object else "building"
        act_lower = claimed_action.lower() if claimed_action else "constructed"

        is_bldg = any(k in obj_lower for k in ["building", "house", "structure", "facility", "shed"])
        is_road = any(k in obj_lower for k in ["road", "lane", "path", "street", "highway"])
        is_const = any(k in act_lower for k in ["constructed", "built", "appeared", "added", "new"])
        is_demo = any(k in act_lower for k in ["demolished", "removed", "cleared", "destroyed"])
        is_unchanged = any(k in act_lower for k in ["unchanged", "no_change", "same", "identical"])

        aspect_ratio = max(h, w) / max(min(h, w), 1)

        visible_change = False
        state_transition = "no_physical_change"
        confidence = 0.50
        observations = ""

        # CASE 0: Claim specifically asserts no change
        if is_unchanged:
            visible_change = False
            state_transition = "no_physical_change"
            confidence = float(np.clip(ssim_score, 0.5, 0.99))
            observations = (
                f"No verifiable physical structural change visible in this crop. "
                f"Structural similarity SSIM={ssim_score:.3f}. Ground cover consistent."
            )

        # CASE 1: Building Construction Claim
        elif is_bldg and is_const:
            # Building appearance requires localized high-contrast anomaly OR clear rectilinear edge appearance
            has_localized_anomaly = (max_block_diff > 82.0 and peak_ratio > 2.05)
            has_edge_increase = (delta_edge > self.structural_density_thresh and crop_mean_diff > 20.0)
            has_rectilinear_appearance = (delta_ortho > 0.10 and ortho2 > 0.50)

            if (has_localized_anomaly or has_edge_increase or has_rectilinear_appearance) and not is_jitter:
                visible_change = True
                state_transition = "appearance_of_structure"
                confidence = float(np.clip(
                    0.5 + 0.25 * min(peak_ratio / 2.5, 1.0) + 0.25 * (1.0 - ssim_score),
                    0.6, 0.98
                ))
                observations = (
                    f"Visible change confirmed. Rectilinear structural boundaries present in T2 "
                    f"(localized max_block={max_block_diff:.1f}, peak/mean ratio={peak_ratio:.2f}, "
                    f"edge density={e2_density:.2f}) that were absent in T1 (edge density={e1_density:.2f}). "
                    f"Visual evidence supports {claimed_object} appearance."
                )
            else:
                visible_change = False
                state_transition = "no_physical_change"
                confidence = float(np.clip(0.6 + 0.3 * (1.0 / max(peak_ratio, 1.0)), 0.6, 0.95))
                observations = (
                    f"No verifiable physical structural change visible in this crop. "
                    f"T1 and T2 display consistent ground texture (delta edge={delta_edge:.3f}, "
                    f"lum shift={lum_shift:+.1f}, max_block={max_block_diff:.1f}, peak_ratio={peak_ratio:.2f}). "
                    f"Any claimed new structure is unsupported."
                )

        # CASE 2: Building Demolition Claim
        elif is_bldg and is_demo:
            has_structure_removed = (
                (delta_edge < -self.structural_density_thresh and e1_density >= self.structural_density_thresh) or
                (delta_ortho < -0.10 and ortho1 > 0.50)
            )
            if has_structure_removed and not is_jitter:
                visible_change = True
                state_transition = "removal_of_structure"
                confidence = float(np.clip(0.5 + 0.5 * abs(delta_edge), 0.6, 0.95))
                observations = (
                    f"Visible change confirmed. Structural edges present in T1 (edge density={e1_density:.2f}) "
                    f"are removed in T2 (edge density={e2_density:.2f}). Cleared or altered surface visible."
                )
            else:
                visible_change = False
                state_transition = "no_physical_change"
                confidence = 0.70
                observations = f"No verifiable physical structural change visible in this crop. T1 structure remains."

        # CASE 3: Road / Infrastructure Corridor Claim
        elif is_road:
            has_corridor = (crop_mean_diff > 25.0 and aspect_ratio > 1.80 and not is_jitter)
            if has_corridor:
                visible_change = True
                state_transition = "surface_alteration"
                confidence = float(np.clip(0.6 + 0.2 * min(aspect_ratio / 3.0, 1.0), 0.6, 0.92))
                observations = (
                    f"Visible surface change observed (mean normalized diff={crop_mean_diff:.1f}, "
                    f"aspect ratio={aspect_ratio:.1f}). Supports {claimed_object} change."
                )
            else:
                visible_change = False
                state_transition = "no_physical_change"
                confidence = 0.70
                observations = f"No road corridor identified in candidate region."

        # CASE 4: Generic Surface / Vegetation Alteration
        else:
            if crop_mean_diff > 25.0 and "shadow_hazard" not in quality_flags and not is_jitter:
                visible_change = True
                state_transition = "surface_alteration"
                confidence = float(np.clip(1.0 - ssim_score, 0.5, 0.90))
                observations = (
                    f"Visible surface change observed (mean diff={crop_mean_diff:.1f}, "
                    f"SSIM={ssim_score:.3f})."
                )
            else:
                visible_change = False
                state_transition = "no_physical_change"
                confidence = float(np.clip(ssim_score, 0.5, 0.95))
                observations = f"No verifiable physical structural change visible in this crop."

        return {
            "visible_change": visible_change,
            "confidence": float(np.clip(confidence, 0.0, 1.0)),
            "state_transition": state_transition,
            "observations": observations,
            "visibility_flags": visibility_flags,
            "quality_flags": quality_flags,
            "structural_edge_change": delta_edge,
            "mean_luminance_diff": lum_shift,
            "crop_mean_diff": crop_mean_diff,
            "ssim_score": ssim_score
        }
