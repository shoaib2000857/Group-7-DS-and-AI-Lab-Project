"""End-to-end TerraWatch Milestone 4 verification pipeline with:
  - Full-data trained BIT detector & RSCaMa captioner (with checkpoint SHA-256 provenance)
  - Contextual crop extraction (min_size=32, padding=0.25, original-T1 half-open coordinates)
  - Linguistic article ('a'/'an') vs explicit numeric count ('three', '2') policy
  - Structural visual witness (OpenCV LAB luminance, Sobel edges, block anomaly, SSIM, jitter)
  - Explicit disagreement/uncertainty triggers + frozen Qwen3-VL-4B-Instruct visual referee
  - Operating-threshold gate & auditable report.json compatible with the Evidence Review UI
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_contextual_bbox(
    bbox: List[int],
    image_shape: Tuple[int, int],
    padding_fraction: float = 0.25,
    min_size: int = 32,
) -> Tuple[List[int], Dict[str, Any]]:
    """Pad candidate bbox with context and guarantee >= min_size (32px >= 24px witness threshold)."""
    H, W = image_shape[:2]
    x0, y0, x1, y1 = [int(v) for v in bbox]
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    pad_w = max(int(round(w * padding_fraction)), 4)
    pad_h = max(int(round(h * padding_fraction)), 4)

    target_w = max(w + 2 * pad_w, min(min_size, W))
    target_h = max(h + 2 * pad_h, min(min_size, H))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    px0 = int(round(cx - target_w / 2.0))
    py0 = int(round(cy - target_h / 2.0))
    px1 = px0 + target_w
    py1 = py0 + target_h

    if px0 < 0:
        px1 = min(W, px1 - px0)
        px0 = 0
    if py0 < 0:
        py1 = min(H, py1 - py0)
        py0 = 0
    if px1 > W:
        px0 = max(0, px0 - (px1 - W))
        px1 = W
    if py1 > H:
        py0 = max(0, py0 - (py1 - H))
        py1 = H

    meta = {
        "original_bbox": [x0, y0, x1, y1],
        "padded_bbox": [px0, py0, px1, py1],
        "padding_applied": {
            "pad_x0": x0 - px0,
            "pad_y0": y0 - py0,
            "pad_x1": px1 - x1,
            "pad_y1": py1 - y1,
        },
        "min_size_enforced": min_size,
        "coordinate_frame": "original_T1_pixels",
        "bbox_convention": "x0_y0_x1_y1_half_open",
    }
    return [px0, py0, px1, py1], meta


def candidate_box_and_components(
    mask: np.ndarray,
    claim_dict: dict,
    min_component_px: int = 16,
) -> Tuple[List[int], bool, Dict[str, Any]]:
    """Locate candidate change region from predicted 0/1/2 class mask."""
    h, w = mask.shape
    obj = claim_dict["claim"]["object"]
    cls = {"road": 1, "building": 2}.get(obj)
    fg = (mask == cls) if cls is not None else (mask > 0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype("uint8"), 8)
    loc = claim_dict["claim"].get("location")
    rows = set(loc["rows"]) if loc else {0, 1, 2}
    cols = set(loc["cols"]) if loc else {0, 1, 2}

    all_comps = []
    matches = []
    for x, y, bw, bh, area in stats[1:]:
        if int(area) < min_component_px:
            continue
        r = min(2, int((y + bh / 2.0) * 3 / h))
        c = min(2, int((x + bw / 2.0) * 3 / w))
        box = [int(x), int(y), int(x + bw), int(y + bh)]
        all_comps.append((int(area), box, (r, c)))
        if r in rows and c in cols:
            matches.append((int(area), box))

    class_px = int(fg.sum())
    any_px = int((mask > 0).sum())
    summary = {
        "target_class": obj,
        "class_change_pixels": class_px,
        "any_change_pixels": any_px,
        "road_pixels": int((mask == 1).sum()),
        "building_pixels": int((mask == 2).sum()),
        "matching_components": len(matches),
        "total_class_components": len(all_comps),
    }
    if matches:
        # Union bounding box of top matching components if multiple exist in the same grid area,
        # or largest matching component
        best_area, best_box = max(matches, key=lambda t: t[0])
        summary["selected_component_area_px"] = best_area
        return best_box, True, summary
    if all_comps and not loc:
        best_area, best_box, _ = max(all_comps, key=lambda t: t[0])
        summary["selected_component_area_px"] = best_area
        return best_box, True, summary
    return [0, 0, w, h], False, summary


def compute_disagreement_triggers(
    claim_dict: dict,
    localized: bool,
    mask_summary: Dict[str, Any],
    witness_obs: Dict[str, Any],
    mask_alarm_px: int = 50,
) -> List[str]:
    """Explicit disagreement/uncertainty trigger policy for M1's targeted reinspection hypothesis."""
    triggers: List[str] = []
    obj = claim_dict["claim"]["object"]
    act = claim_dict["claim"]["action"]
    parser_meta = claim_dict.get("parser", {})
    qty_kind = parser_meta.get("quantity_kind")

    if act == "no_change":
        if mask_summary["any_change_pixels"] >= mask_alarm_px:
            triggers.append("T_mask_caption_conflict_no_change")
    elif act in {"added", "removed", "modified", "changed"}:
        if obj in {"building", "road"} and not localized:
            triggers.append("T_no_matching_mask_region")
        if obj in {"building", "road"} and mask_summary["class_change_pixels"] < mask_alarm_px:
            triggers.append("T_weak_mask_support")
        if witness_obs.get("visible_change") is False:
            triggers.append("T_witness_disagrees_no_visual_change")
    if witness_obs.get("visible_change") is None:
        triggers.append("T_witness_indeterminate")
    q_flags = witness_obs.get("quality_flags", [])
    if any("shadow_hazard" in f or "jitter" in f for f in q_flags):
        triggers.append("T_quality_or_alignment_hazard")
    if qty_kind == "exact":
        triggers.append("T_exact_numeric_count")
    if parser_meta.get("hedged"):
        triggers.append("T_hedged_claim")
    if obj not in {"building", "road", "scene", "region"}:
        triggers.append("T_out_of_mask_ontology")
    return triggers


def mask_only_decision(claim_dict: dict, localized: bool, mask_summary: Dict[str, Any], alarm_px: int = 50) -> str:
    """Setting 3 baseline (M1): check claim solely against the predicted BIT mask."""
    obj = claim_dict["claim"]["object"]
    act = claim_dict["claim"]["action"]
    if act == "insufficient_evidence":
        return "defer"
    if act == "no_change":
        return "accept" if mask_summary["any_change_pixels"] < alarm_px else "reject"
    if obj == "region":
        return "accept" if mask_summary["any_change_pixels"] >= alarm_px else "reject"
    if obj not in {"building", "road"}:
        return "defer"
    return "accept" if (localized and mask_summary["class_change_pixels"] >= alarm_px) else "reject"


def apply_decision_gate(
    raw: Dict[str, Any],
    threshold: float,
    forced_defer_reasons: List[str],
) -> Dict[str, Any]:
    """Deterministic safety gate applied after Qwen3-VL referee readout."""
    reasons = list(forced_defer_reasons)
    outcome = raw.get("outcome", "defer")
    probs = raw.get("probabilities")
    if outcome not in {"accept", "reject", "defer"}:
        reasons.append("invalid_decision")
        outcome = "defer"
    if outcome == "accept":
        valid = isinstance(probs, dict) and set(probs) == {"accept", "reject", "defer"}
        valid = valid and all(
            isinstance(v, (int, float)) and math.isfinite(v) and 0.0 <= v <= 1.0
            for v in probs.values()
        )
        valid = valid and math.isclose(sum(probs.values()), 1.0, abs_tol=0.02)
        if not valid:
            reasons.append("unverified_accept_probability")
        elif probs["accept"] < threshold:
            reasons.append("accept_below_operating_threshold")
    return {
        "outcome": "defer" if reasons else outcome,
        "raw_outcome": raw.get("outcome"),
        "probabilities": probs,
        "reason_codes": reasons,
        "raw_response": raw.get("raw"),
        "probability_semantics": raw.get("probability_semantics"),
        "accept_threshold": threshold,
    }


def make_mask_overlay(t2_rgb: np.ndarray, mask: np.ndarray, bbox: Optional[List[int]] = None) -> np.ndarray:
    """Create human-readable overlay (road=yellow, building=red) with optional candidate box."""
    vis = t2_rgb.copy().astype(np.float32)
    road = mask == 1
    bldg = mask == 2
    if road.any():
        vis[road] = vis[road] * 0.45 + np.array([255, 220, 40], dtype=np.float32) * 0.55
    if bldg.any():
        vis[bldg] = vis[bldg] * 0.45 + np.array([235, 55, 55], dtype=np.float32) * 0.55
    out = np.clip(vis, 0, 255).astype(np.uint8)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 160), 2)
    return out


REFEREE_INSTRUCTIONS = (
    "You are an independent visual verifier inspecting two aligned overhead satellite images of the same location. "
    "Image 1 is earlier (T1) and Image 2 is later (T2). When zoomed crops are also provided, Image 3 is the zoomed T1 "
    "candidate region and Image 4 is the zoomed T2 candidate region. "
    "Judge whether the claim is visually supported by the actual images. Treat detector masks and structural witness "
    "statistics as fallible hints, NOT ground truth. "
    "Accept (A) when the images clearly visually support the claim (including clearly unchanged scenes for 'no change' claims). "
    "Reject (B) when the images contradict the claim (e.g., claiming new buildings where none appeared, or claiming no change where clear buildings/roads appeared). "
    "Defer (C) when visual evidence is ambiguous, blurry, too small to tell, or asserts exact counts/non-visual facts. "
    "Respond with exactly one letter: A, B, or C."
)


class Qwen3VLReferee:
    """Pinned Qwen3-VL-4B-Instruct referee supporting direct full-image and crop+context reinspection."""

    OUTCOMES = ["accept", "reject", "defer"]

    def __init__(
        self,
        repo: str = "Qwen/Qwen3-VL-4B-Instruct",
        revision: str = "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        device: str = "cuda",
        max_image_side: int = 512,
    ):
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.max_image_side = max_image_side
        dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
        self.processor = AutoProcessor.from_pretrained(
            repo, revision=revision, min_pixels=65536, max_pixels=max_image_side ** 2
        )
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            repo, revision=revision, dtype=dtype, device_map=device, low_cpu_mem_usage=True
        ).eval()
        self.code_ids = [self.tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABC"]
        self.metadata = {
            "repo": repo,
            "revision": revision,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "dtype": str(dtype),
        }

    def decide(self, state: Dict[str, Any], image_paths: List[str], mode: str = "reinspection") -> Dict[str, Any]:
        choices = (
            "A: The images visually support the claim.\n"
            "B: The images visually contradict the claim.\n"
            "C: Insufficient or ambiguous visual evidence, or outside visual scope."
        )
        prompt = (
            f"{REFEREE_INSTRUCTIONS}\n"
            f"Verification mode: {mode}\n"
            f"Claim and structured context:\n{json.dumps(state, ensure_ascii=True)}\n"
            f"{choices}\n"
            "Respond with exactly one code: A, B or C."
        )
        images = []
        for p in image_paths:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
            if max(rgb.size) < 128:
                # Upsample small crops cleanly for VLM patch visibility while preserving original saved crop on disk
                scale = max(2, 128 // max(1, min(rgb.size)))
                rgb = rgb.resize((rgb.size[0] * scale, rgb.size[1] * scale), Image.Resampling.BICUBIC)
            rgb.thumbnail((self.max_image_side, self.max_image_side))
            images.append(rgb)

        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**inputs, logits_to_keep=1).logits[0, -1, self.code_ids].float()
            probs_list = self.torch.softmax(logits, dim=-1).cpu().tolist()
        probs = dict(zip(self.OUTCOMES, probs_list))
        outcome = max(probs, key=probs.get)
        return {
            "outcome": outcome,
            "probabilities": probs,
            "probability_semantics": "softmax_conditional_on_three_label_tokens_not_calibrated_truth",
            "raw": {
                "restricted_codes": dict(zip("ABC", probs_list)),
                "code_order": self.OUTCOMES,
                "input_tokens": int(inputs["input_ids"].shape[-1]),
                "num_images": len(images),
                "mode": mode,
            },
        }
