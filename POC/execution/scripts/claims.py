"""Deterministic claim extraction for change captions (B deliverable 2).

caption -> atomic claims {object, action, location, quantity, quantity_unit}
with exact source spans and character offsets.

rules-v3 (frozen 27 Oct 2026). Developed on the 50 development pairs' reference
captions and SmolVLM2/Qwen3-VL development outputs. Disclosure: screening
outputs also shaped two revisions. v2 (increase-in, not-present-in-T1, copula
"appears to be", landmark nouns, "multiple lanes") came from reading 12
SmolVLM2 screening outputs; v3 (bare "the same", "yard/driveway of", vehicle
"on the street", "now occupied by") came from an independent spot check of
36 screening claims and one data-story example.

Never corrects the text: qualifiers ("possibly"), negation and vague
quantities are kept as parser metadata, not upgraded to exact claims.
Prose locations are mapped to a 3x3 image grid; they do not establish pixel
coordinates, so candidate_region_ids stays empty.
"""
import re

BUILDING = r"buildings?|houses?|villas?|bungalows?|mansions?|residential (?:area|houses|buildings|district|block)s?|residences?|homes?|structures?|apartments?|cottages?|factor(?:y|ies)|warehouses?|sheds?|rooftops?|roofs?|constructions?|facilit(?:y|ies)|dwellings?|construction sites?"
ROAD = r"roads?|paths?|streets?|crossroads?|cross roads?|highways?|tracks?|intersections?|driveways?|roadways?|pavements?|avenues?|bypass(?:es)?|junctions?"
OTHER = {
    "vegetation": r"trees?|vegetation|plants?|grass(?:land)?|woods?|woodland|forests?|shrubs?|bushes?|meadows?|greenery|lawns?|plantation",
    "water": r"lakes?|pools?|ponds?|rivers?|water|swimming pools?",
    "parking": r"parking (?:lot|place|space|area)s?|car parks?",
    "vehicle": r"vehicles?|cars?|trucks?",
    "bareland": r"bare ?land|bare soil|desert|wasteland|open space|clearing|sand|soil|land",
}
REGION = r"changed regions?|changed areas?|areas? of change"
OBJECT_PATTERNS = [("region", REGION), ("building", BUILDING), ("road", ROAD)] + list(OTHER.items())
OBJ_RE = re.compile(
    "|".join(f"(?P<{name}>\\b(?:{pat})\\b)" for name, pat in OBJECT_PATTERNS), re.I
)

COPULA_AFTER = r"(?!\s+(?:to|more|less|larger|smaller|bigger|wider|narrower|denser|darker|lighter|greener|different|similar|consistent|unchanged|the same|in the same|identical|largely|mostly|relatively|slightly|very|quite|somewhat|well|clearer|sharper))"
ADD = r"appear(?:s|ed|ing)?" + COPULA_AFTER + r"|increase[sd]? in (?:the )?(?:number|density|amount|count)|additional|(?:(?:is|are|was|were) )?not (?:present|visible|there|seen) in (?:image )?t1\b|(?:was|were) not (?:present|visible|there|seen)(?! in (?:image )?t2)|did not exist|now (?:occupied|covered|filled|dominated) (?:by|with)|(?:been )?occupied by|absent (?:in|from) (?:image )?t1|more(?=\s+(?:\w+\s+)?(?:houses|buildings|roads|structures|villas|homes|residences))|(?:been |being |is |are |was |were )?(?:built|constructed|added|erected|developed|paved|laid|planted)|new(?:ly)?|emerge[sd]?|show(?:s|ed)? up|ha(?:s|ve) appeared|construction of|addition of|there (?:is|are) now|now (?:has|have|contains?)|replac(?:e|es|ing)"
REMOVE = r"(?:(?:is|are|was|were) )?(?:not|no longer) (?:present|visible|there) in (?:image )?t2|absent (?:in|from) (?:image )?t2|missing (?:in|from) (?:image )?t2|decrease[sd]? in (?:the )?(?:number|density)|disappear(?:s|ed|ing)?|(?:been |being |is |are |was |were )?(?:removed|demolished|destroyed|cleared|cut|torn down|razed)|vanish(?:es|ed)?|gone|no longer|missing|replaced by|replaced with|is replaced|are replaced|removal of|demolition of"
MODIFY = r"changes? in (?:the )?(?:\w+'s )?(?:location|layout|position|surface|shape|size|structure|appearance|design|colou?r|roof|width|alignment|configuration|arrangement)|more (?:developed|built[- ]up|organi[sz]ed|structured)|becomes? (?:greener|lusher|denser|wider|larger|bigger|thicker|more \\w+)|widen(?:s|ed)?|wider|expand(?:s|ed)?|extended|enlarged|rebuilt|reconstructed|renovated|reshaped|resurfaced|upgraded|modified|changed into|become[s]?|turned into|developed into|converted"
ACTION_RES = [("removed", re.compile(rf"\b(?:{REMOVE})\b", re.I)),
              ("modified", re.compile(rf"\b(?:{MODIFY})\b", re.I)),
              ("added", re.compile(rf"\b(?:{ADD})\b", re.I))]

NO_CHANGE_RE = re.compile(
    r"\bno candidate changed regions?\b|"
    r"\bno (?:other |visible |relevant |significant |obvious |noticeable |apparent |discernible |major |further )*(?:difference|change)s?\b|"
    r"\bno (?:other |visible |relevant |significant |obvious |noticeable |apparent |discernible |major |new |further )*(?:building|road|structural|construction)s?(?: or (?:building|road)s?)? (?:changes?|differences?)\b|"
    r"\b(?:identical|unchanged|same as before|(?:is|are|remains?|remained|looks?|seems?|stays?|stayed) (?:largely |mostly |essentially )?the same|nothing (?:has )?changed|almost nothing|nothing new|"
    r"remain(?:s|ed)? (?:the same|unchanged|largely unchanged)|no (?:new )?(?:buildings?|roads?) (?:have|has|were|are|was)? ?(?:been )?(?:built|added|constructed|removed|changed))\b",
    re.I,
)
INSUFFICIENT_RE = re.compile(
    r"\b(?:insufficient|not enough|cannot (?:be )?(?:determined|confirmed|seen)|unclear|difficult to (?:determine|tell|see)|hard to tell)\b",
    re.I,
)
HEDGE_RE = re.compile(r"\b(?:possibly|perhaps|may|might|likely|appears? to be|seems?|probably|could|potential(?:ly)?|suggest(?:s|ing)?)\b", re.I)
NEG_RE = re.compile(r"\b(?:no|not|none|without|neither|nor)\b", re.I)

NUMBERS = {"one": 1, "a": 1, "an": 1, "single": 1, "two": 2, "three": 3, "four": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "a pair of": 2, "both": 2, "another": 1}
VAGUE = {"some": "few", "several": "few", "a few": "few", "few": "few", "a couple of": "few",
         "a small number of": "few", "many": "many", "lots of": "many", "a lot of": "many",
         "massive": "many", "numerous": "many", "multiple": "many", "rows of": "many",
         "a row of": "many", "dozens of": "many", "a large number of": "many", "most": "many",
         "a cluster of": "many", "clusters of": "many", "a group of": "many", "groups of": "many"}
QUANT_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, list(NUMBERS) + list(VAGUE)), key=len, reverse=True))
    + r"|\d+)\b(?=(?:\s+\w+){0,3}\s*$)", re.I)

ROW = {"top": 0, "upper": 0, "north": 0, "northern": 0, "above": 0,
       "middle": 1, "center": 1, "centre": 1, "central": 1,
       "bottom": 2, "lower": 2, "south": 2, "southern": 2, "below": 2}
COL = {"left": 0, "west": 0, "western": 0, "middle": 1, "center": 1, "centre": 1,
       "central": 1, "right": 2, "east": 2, "eastern": 2}
LOC_RE = re.compile(
    r"\b(?:(?:at|in|on|near|along|toward|towards|across|to|of) (?:the )?)?"
    r"(?:(?:top|upper|bottom|lower|middle|center|centre|central|north|south|northern|southern)"
    r"(?:[- ](?:left|right|middle|center|centre|half|part|portion|area|side|edge|corner|section|region))*"
    r"|left|right|west|east|western|eastern|corners?|edges?|border)"
    r"(?:[- ](?:hand|half|part|portion|area|side|edge|corner|section|region|quadrant|of the (?:image|scene|picture|area)))*\b",
    re.I,
)
SPLIT_RE = re.compile(r",\s*(?:and\s+|while\s+|but\s+)?|\s+(?:and|while|but|whereas|as well as|with)\s+|;\s*")


def norm_location(text):
    """Map image-relative prose to sets of 3x3 grid rows/cols (None = unconstrained)."""
    t = text.lower().replace("-", " ")
    words = re.findall(r"[a-z]+", t)
    rows = {ROW[w] for w in words if w in ROW and w not in ("middle", "center", "centre", "central")}
    cols = {COL[w] for w in words if w in COL and w not in ("middle", "center", "centre", "central")}
    centred = any(w in ("middle", "center", "centre", "central") for w in words)
    if centred:
        if not rows and not cols:
            rows, cols = {1}, {1}
        elif rows and not cols:
            cols = {1}
        elif cols and not rows:
            rows = {1}
    if ("corner" in words or "corners" in words) and not rows and not cols:
        rows, cols = {0, 2}, {0, 2}
    if any(w in words for w in ("edge", "edges", "border")) and not rows and not cols:
        rows, cols = {0, 2}, {0, 1, 2}
    if not rows and not cols:
        return None
    return {"rows": sorted(rows) if rows else [0, 1, 2],
            "cols": sorted(cols) if cols else [0, 1, 2]}


def _quantity(prefix):
    m = None
    for m in QUANT_RE.finditer(prefix):
        pass
    if not m:
        return None
    w = m.group(0).lower()
    if w.isdigit():
        return {"value": int(w), "kind": "exact", "text": m.group(0)}
    if w in NUMBERS:
        return {"value": NUMBERS[w], "kind": "exact" if w not in ("a", "an", "another") else "article",
                "text": m.group(0)}
    return {"value": VAGUE[w], "kind": "vague", "text": m.group(0)}


def _segments(sentence, offset):
    """Split a sentence into coordinated segments, keeping absolute offsets."""
    pos, out = 0, []
    for m in SPLIT_RE.finditer(sentence):
        out.append((sentence[pos:m.start()], offset + pos))
        pos = m.end()
    out.append((sentence[pos:], offset + pos))
    return [(s, o) for s, o in out if s.strip()]


def _actions(text):
    found = []
    for name, rx in ACTION_RES:
        for m in rx.finditer(text):
            found.append((m.start(), name, m.group(0)))
    return sorted(found)


def extract_claims(caption, pair_id=None, run_id=None):
    """Return a list of claim dicts with exact spans into `caption`."""
    claims = []
    if not caption:
        return claims
    for sm in re.finditer(r"[^.!?]+[.!?]?", caption):
        sent, s0 = sm.group(0), sm.start()
        if not sent.strip():
            continue
        hedged = bool(HEDGE_RE.search(sent))
        if INSUFFICIENT_RE.search(sent):
            claims.append(_claim(caption, s0, s0 + len(sent), "scene", "insufficient_evidence",
                                 None, None, hedged, False))
            continue
        # Clauses: split on commas/semicolons/contrast words, not on "and",
        # so "no building or road changes" stays one scope.
        for clause, c0 in _clauses(sent, s0):
            if NO_CHANGE_RE.search(clause):
                scoped = sorted({m.lastgroup for m in OBJ_RE.finditer(clause)
                                 if m.lastgroup in ("building", "road")})
                for obj in scoped or ["scene"]:
                    claims.append(_claim(caption, c0, c0 + len(clause), obj, "no_change",
                                         _loc(clause), None, hedged, True))
                continue
            claims.extend(_change_claims(caption, clause, c0, hedged))
    return _dedupe(claims, pair_id, run_id)


CLAUSE_RE = re.compile(r",\s*(?:and\s+|while\s+|but\s+)?|;\s*|\s+(?:but|while|whereas|although|however)\s+")


def _clauses(sentence, offset):
    pos, out = 0, []
    for m in CLAUSE_RE.finditer(sentence):
        out.append((sentence[pos:m.start()], offset + pos))
        pos = m.end()
    out.append((sentence[pos:], offset + pos))
    merged = []
    for c, o in out:
        if not c.strip():
            continue
        # object-less fragments ("bottom and lower-right of the image") belong to the previous clause
        if merged and not OBJ_RE.search(c) and not NO_CHANGE_RE.search(c):
            pc, po = merged[-1]
            merged[-1] = (sentence[po - offset:o - offset + len(c)], po)
        else:
            merged.append((c, o))
    return merged


def _change_claims(caption, clause, c0, hedged):
    claims = []
    seg_info = []
    for seg, o in _segments(clause, c0):
        seg_info.append((seg, o, _actions(seg), list(OBJ_RE.finditer(seg))))
    for idx, (seg, o, acts, objs) in enumerate(seg_info):
        if not objs:
            continue
        # Inherit an action from a later segment ("a lake and a building have appeared"),
        # else from an earlier one.
        inherited_into = replaced_before = False
        if not acts:
            donor = next((s for s in seg_info[idx + 1:] if s[2]), None) or \
                next((s for s in reversed(seg_info[:idx]) if s[2]), None)
            acts = donor[2] if donor else []
            # "X becomes a parking lot with a house": things attached to the result are new.
            inherited_into = bool(donor and donor[1] < o and REPLACE_INTO_RE.search(donor[0]))
            # "X previously ... and [it] is now occupied by Y": X is what was replaced
            replaced_before = bool(donor and donor[1] > o and REPLACE_INTO_RE.search(donor[0]))
        if not acts and any(om.lastgroup == "region" for om in objs):
            acts = [(0, "changed", "")]  # "candidate changed region at ...": the noun carries the change
        if not acts:
            continue
        tail = seg
        for nxt in seg_info[idx + 1:]:
            if nxt[3]:
                break
            tail += " " + nxt[0]
        loc = _loc(tail)
        negated = False
        for om in objs:
            obj = om.group(0)
            cls = om.lastgroup
            action = "added" if inherited_into else ("removed" if replaced_before else _action_for(seg, om, acts))
            if cls == "region":
                action = "changed"
            if cls == "bareland" and action in ("added", "modified"):
                continue  # "built on the bareland": land is the site, not a new object
            q = _quantity(seg[:om.start()] + " " + obj)
            negated = bool(NEG_RE.search(seg[:om.start()]))
            if negated:
                action = "no_change"
            if LANDMARK_RE.search(seg[:om.start()]):
                continue  # "on the left side of the road": the road is a reference point, not a claim
            if cls != "vehicle" and VEHICLE_BEFORE_RE.search(seg[:om.start()]):
                continue  # "a red car appears on the street": the street is where the car is
            claims.append(_claim(caption, o + om.start(), o + om.end(), cls, action, loc, q,
                                 hedged, negated, seg_span=(o, o + len(seg)), noun=obj.lower()))
    return claims


LANDMARK_RE = re.compile(
    r"\b(?:(?:side|sides|edge|edges|end|part|corner|middle|centre|center|length|yard|yards|driveway|driveways|roof|rooftop|front|back|rear|lot|grounds?|perimeter|entrance|edge) of|in front of|behind|outside|along|alongside|beside|"
    r"besides|near|next to|adjacent to|close to|across|around|lining|lines|facing|onto|toward|towards|"
    r"(?:on|at|in|to) (?:the )?(?:(?:left|right|north|south|east|west|other|either|opposite|far) )?(?:side|sides) of)"
    r"\s+(?:the |a |an |this |that |these |those |both |either |each |main |existing |original |former |same )*"
    r"(?:\w+\s+)?$", re.I)


VEHICLE_BEFORE_RE = re.compile(
    r"\b(?:" + OTHER["vehicle"] + r")\b[^.;]*?\b(?:on|in|at|along|onto|into|near|across|inside)\s+"
    r"(?:the |a |an |this |that |same |main |nearby |adjacent )*(?:\w+\s+)?$", re.I)


REPLACE_INTO_RE = re.compile(r"\b(?:(?:now |been )?(?:occupied|covered|filled|dominated|taken over) (?:by|with)|replaced (?:by|with)|turned into|developed into|changed into|converted (?:in)?to|becomes? (?:a|an|the|some|many|two|several)\b)", re.I)


def _action_for(seg, om, acts):
    """Pick the action governing an object; handles 'X replaced by Y' / 'Y replaces X'."""
    lower = seg.lower()
    rb = re.search(r"\b(?:now (?:occupied|covered|filled|dominated) (?:by|with)|(?:been )?(?:occupied|taken over) by|replaced (?:by|with)|turned into|developed into|changed into|converted (?:in)?to|becomes? (?:a|an|the|some|many|two|several)\b|gives way to)\b", lower)
    if rb:
        return "removed" if om.start() < rb.start() else "added"
    rp = re.search(r"\b(?:replac(?:e|es|ing)|in place of|instead of)\b", lower)
    if rp:
        return "added" if om.start() < rp.start() else "removed"
    if re.search(r"\bon (?:the )?(?:original |former )?" + re.escape(om.group(0).lower()), lower):
        return "site"
    # nearest action mention, preferring the first after the object (passive / verb-final)
    after = [a for a in acts if a[0] >= om.end() - 0] if acts and acts[0][0] >= 0 else []
    pick = (after or acts)[0] if acts else None
    if pick is None:
        return None
    return pick[1]


OF_LANDMARK_RE = re.compile(r"\s+of\s+(?:the\s+|a\s+|an\s+)?(?:\w+\s+)?(?:" + BUILDING + "|" + ROAD + "|" + "|".join(OTHER.values()) + r")\b", re.I)


def _loc(text):
    """Union of all image-relative locations named in the text."""
    texts, cells = [], set()
    for m in LOC_RE.finditer(text):
        if OF_LANDMARK_RE.match(text, m.end()):
            continue  # "right side of the road" is relative to an object, not the image
        grid = norm_location(m.group(0))
        if grid:
            texts.append(m.group(0).strip())
            cells |= {(r, c) for r in grid["rows"] for c in grid["cols"]}
    if cells:
        return {"text": "; ".join(texts), "rows": sorted({r for r, _ in cells}),
                "cols": sorted({c for _, c in cells}), "cells": sorted(cells)}
    if re.search(r"\bboth sides\b|\balong\b|\bbeside\b|\bnext to\b|\bnear\b|\baround\b", text, re.I):
        return {"text": "relational", "rows": [0, 1, 2], "cols": [0, 1, 2], "relational": True}
    return None


def _claim(caption, a, b, cls, action, loc, q, hedged, negated, seg_span=None, noun=None):
    span = seg_span or (a, b)
    return {
        "source_span": caption[span[0]:span[1]].strip(),
        "char_start": span[0], "char_end": span[1],
        "claim": {
            "object": cls,
            "object_text": noun,
            "action": action,
            "location": loc,
            "quantity": None if q is None else q["value"],
            "quantity_unit": None if q is None else noun,
        },
        "parser": {"quantity_kind": None if q is None else q["kind"],
                   "quantity_text": None if q is None else q["text"],
                   "hedged": hedged, "negated": negated, "version": "rules-v3"},
    }


def _dedupe(claims, pair_id, run_id):
    seen, out = set(), []
    for c in claims:
        if c["claim"]["action"] == "site":
            continue
        key = (c["claim"]["object"], c["claim"]["action"],
               tuple(c["claim"]["location"]["rows"]) if c["claim"]["location"] else None,
               tuple(c["claim"]["location"]["cols"]) if c["claim"]["location"] else None)
        if key in seen:
            continue
        seen.add(key)
        c["claim_id"] = f"{run_id}:{pair_id}:c{len(out)}" if run_id else f"c{len(out)}"
        c["pair_id"], c["run_id"] = pair_id, run_id
        out.append(c)
    return out


def to_contract(c):
    """Shape a claim like templates/claim_decision.json (evidence/decision left unpopulated)."""
    loc = c["claim"]["location"]
    return {
        "record_type": "claim_from_caption",
        "pair_id": c["pair_id"], "claim_id": c["claim_id"], "source_span": c["source_span"],
        "claim": {"object": c["claim"]["object"], "action": c["claim"]["action"],
                  "location": None if loc is None else loc["text"],
                  "quantity": c["claim"]["quantity"], "quantity_unit": c["claim"]["quantity_unit"]},
        "candidate_region_ids": [],
        "coordinate_frame": "original_T1_pixels", "bbox_convention": "x0_y0_x1_y1_half_open",
        "evidence": {"mask_run_id": None, "mask_ontology": None, "witness_run_id": None,
                     "observations": None, "visibility_flags": [], "quality_flags": [],
                     "crop_paths": [], "transform_record": None},
        "decision": {"outcome": None, "allowed_outcomes": ["accept", "reject", "defer"],
                     "reason_codes": [], "probabilities": None, "probability_semantics": None,
                     "raw_response_path": None, "parse_status": None},
        "human_reference": "stored_separately_and_never_passed_to_inference",
        "b_source": {"run_id": c["run_id"], "char_start": c["char_start"], "char_end": c["char_end"],
                     "location_grid": loc, "parser": c["parser"]},
    }
