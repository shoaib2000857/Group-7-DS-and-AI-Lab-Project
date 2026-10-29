"""Reference-check labels on synthetic masks and references."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate as ev  # noqa: E402
from claims import extract_claims  # noqa: E402


class FakeTruth(ev.Truth):
    def __init__(self, masks, refs):
        self._masks = masks
        self.refs = refs
        self.ref_claims = {p: [c for cap in v["captions"] for c in extract_claims(cap)] for p, v in refs.items()}


def build():
    m_build = np.zeros((256, 256), np.uint8)
    m_build[10:60, 10:60] = 255           # building change, upper-left
    m_empty = np.zeros((256, 256), np.uint8)
    m_road = np.zeros((256, 256), np.uint8)
    m_road[200:220, :] = 128              # road change, bottom
    return FakeTruth(
        {"p/b": m_build, "p/e": m_empty, "p/r": m_road},
        {"p/b": {"changeflag": 1, "captions": ["a building appears at the top left ."]},
         "p/e": {"changeflag": 0, "captions": ["there is no difference ."]},
         "p/r": {"changeflag": 1, "captions": ["a road is built at the bottom ."]}},
    )


def label(T, pid, text):
    return [ev.check_claim(c, pid, T) for c in extract_claims(text, pid, "r")]


def test_supported_with_location():
    T = build()
    (r,) = label(T, "p/b", "A building is built in the upper-left.")
    assert r["label"] == "supported" and r["location_check"] is True


def test_wrong_location_flagged_but_type_supported():
    T = build()
    (r,) = label(T, "p/b", "A building is built in the lower-right.")
    assert r["label"] == "supported" and r["location_check"] is False


def test_false_change_on_unchanged_pair():
    T = build()
    (r,) = label(T, "p/e", "Several houses are constructed in the centre.")
    assert r["label"] == "contradicted"


def test_direction_conflict():
    T = build()
    (r,) = label(T, "p/b", "A building is demolished at the top left.")
    assert r["label"] == "direction_conflict"


def test_no_change_claims():
    T = build()
    assert label(T, "p/e", "There is no difference.")[0]["label"] == "supported"
    assert label(T, "p/r", "No visible change.")[0]["label"] == "contradicted"


def test_wrong_object_type():
    T = build()
    (r,) = label(T, "p/r", "A new building appears at the bottom.")
    assert r["label"] == "contradicted"


def test_out_of_scope_is_unverifiable():
    T = build()
    (r,) = label(T, "p/r", "Trees are removed.")
    assert r["label"] == "unverifiable"
