# Model interfaces

- Inputs: ordered, aligned RGB T1/T2 images with matching dimensions. References are excluded from worker requests.
- Detector: original-frame categorical mask, class 0 background, 1 road, 2 building.
- Captioner: caption text and model/checkpoint metadata.
- Claims: exact caption spans, object/action/location/quantity and parser provenance.
- Evidence: half-open original-T1 pixel boxes, context crops, transform records, quality flags and structural observations.
- Referee: accept/reject/defer, raw response and conditional label-token probabilities. These scores are not calibrated truth probabilities.
- Output: report.json with input hashes, model metadata, accepted-description text, claim decisions, errors and review flags. Existing output directories are preserved.

The executable record schemas are in `src/evidence/schemas.py`. Evaluation reads reference captions and source targets separately from inference.
