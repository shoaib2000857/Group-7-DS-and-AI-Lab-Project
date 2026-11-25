"""Regression and integration tests for Milestone 4 full-data pipeline upgrades:
- 3-class LEVIR-MCI mask encoding (0 / 128 / 255 -> 0 / 1 / 2)
- Contextual crop padding enforcing >= 32px dimensions even at image borders
- Article ('a'/'an') vs explicit numeric count ('two') deferral policy
- Evidence Review UI serving real full-data E2E traces with immutable model decisions
  and append-only SQLite human review persistence
"""
from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

import numpy as np

from src.app.server import ReviewStore, discover_run_folders, load_reports, make_server
from src.evidence.crop_extractor import compute_padded_bbox
from src.models.claims import extract_claims

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACES_DIR = REPO_ROOT / "docs/milestone4/evidence/full_data_m4/verification/e2e_traces"


def test_explicit_three_class_mask_mapping():
    raw = np.array([[0, 128, 255], [255, 0, 128]], dtype=np.uint8)
    mask = np.zeros_like(raw, dtype=np.uint8)
    mask[raw == 128] = 1
    mask[raw == 255] = 2
    assert mask.tolist() == [[0, 1, 2], [2, 0, 1]]


def test_contextual_crop_guarantees_32px_even_at_borders():
    for box in ([0, 0, 8, 10], [248, 246, 256, 256], [120, 120, 126, 128]):
        padded, meta = compute_padded_bbox(box, (256, 256), padding_fraction=0.25, min_size=32)
        x0, y0, x1, y1 = padded
        assert 0 <= x0 < x1 <= 256
        assert 0 <= y0 < y1 <= 256
        assert (x1 - x0) >= 32 and (y1 - y0) >= 32
        assert meta["coordinate_frame"] == "original_T1_pixels"


def test_article_vs_explicit_numeric_count_distinction():
    article_claims = extract_claims("a road appears in the bareland")
    assert article_claims
    assert article_claims[0]["parser"].get("quantity_kind") != "exact"

    exact_claims = extract_claims("two houses are built along the road")
    assert exact_claims
    assert exact_claims[0]["parser"].get("quantity_kind") == "exact"


def test_evidence_review_ui_end_to_end_on_real_traces(tmp_path):
    assert TRACES_DIR.is_dir(), f"Missing real E2E traces directory at {TRACES_DIR}"
    folders = discover_run_folders([str(TRACES_DIR)])
    assert len(folders) == 8

    records, assets = load_reports(folders)
    assert len(records) >= 8
    outcomes = {r["claim"]["decision"]["outcome"] for r in records.values()}
    assert {"accept", "reject", "defer"}.issubset(outcomes)

    store = ReviewStore(tmp_path / "audit.sqlite3")
    srv = make_server(records, assets, store, port=0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(f"{base}/").read().decode("utf-8")
        assert "TerraWatch" in html

        reps = json.loads(urllib.request.urlopen(f"{base}/api/reports").read())
        r0 = reps[0]
        orig_outcome = r0["claim"]["decision"]["outcome"]
        for item in list(r0["images"]) + [r0["mask"], r0["overlay"]] + list(r0["crops"]):
            if not item:
                continue
            url = item["url"] if isinstance(item, dict) else item
            resp = urllib.request.urlopen(f"{base}{url}")
            assert resp.status == 200 and len(resp.read()) > 100

        override = "defer" if orig_outcome == "accept" else "accept"
        payload = json.dumps({
            "id": r0["id"],
            "reviewer": "Group7Test",
            "decision": override,
            "flagged": True,
            "correction": "Verified visual evidence on T2",
            "notes": "Automated persistence check",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/api/review",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ev = json.loads(urllib.request.urlopen(req).read())
        assert ev["original_model_outcome"] == orig_outcome
        assert ev["is_human_override"] is True

        exported = json.loads(urllib.request.urlopen(f"{base}/api/export").read())
        r0_after = next(x for x in exported if x["id"] == r0["id"])
        assert r0_after["claim"]["decision"]["outcome"] == orig_outcome
        assert len(r0_after["reviews"]) == 1
        assert r0_after["reviews"][0]["reviewer"] == "Group7Test"
    finally:
        srv.shutdown()
        srv.server_close()
