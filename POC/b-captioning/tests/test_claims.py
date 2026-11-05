import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from claims import extract_claims, norm_location, to_contract  # noqa: E402


def triples(text):
    return [(c["claim"]["object"], c["claim"]["action"], c["claim"]["quantity"])
            for c in extract_claims(text)]


def test_simple_addition_with_location():
    cs = extract_claims("a building appears on the upper left of the scene .")
    assert len(cs) == 1
    c = cs[0]["claim"]
    assert (c["object"], c["action"], c["quantity"]) == ("building", "added", 1)
    assert c["location"]["cells"] == [(0, 0)]


def test_no_change_variants():
    for t in ["there is no difference .", "the two scenes seem identical .",
              "No visible building or road changes between T1 and T2.",
              "No candidate changed region was detected."]:
        acts = {a for _, a, _ in triples(t)}
        assert acts == {"no_change"}, t


def test_no_change_scope_does_not_leak():
    got = triples("No new buildings were added, but a road was widened at the top.")
    assert ("building", "no_change", None) in got
    assert ("road", "modified", 1) in got
    assert ("road", "no_change", None) not in got


def test_shared_verb_and_replacement():
    got = triples("a lake and a building have appeared to replace the lush trees .")
    assert ("water", "added", 1) in got and ("building", "added", 1) in got
    assert ("vegetation", "removed", None) in got


def test_replaced_by_direction():
    got = triples("plants in the lower area are replaced by a road .")
    assert ("vegetation", "removed", None) in got and ("road", "added", 1) in got


def test_becomes_adjective_is_not_removal():
    got = triples("the grass becomes greener .")
    assert got == [("vegetation", "modified", None)]


def test_site_noun_is_not_a_claim():
    got = triples("many houses are constructed on the bareland .")
    assert got == [("building", "added", "many")]


def test_hedge_and_vague_quantity_preserved():
    c = extract_claims("several new buildings have possibly been constructed in the lower-left .")[0]
    assert c["parser"]["hedged"] is True
    assert c["claim"]["quantity"] == "few" and c["parser"]["quantity_kind"] == "vague"


def test_exact_spans():
    text = "Some text. Two houses are built on the right side."
    for c in extract_claims(text):
        assert text[c["char_start"]:c["char_end"]].strip() == c["source_span"]


def test_location_union_and_centre():
    assert norm_location("in the center") == {"rows": [1], "cols": [1]}
    c = extract_claims("Candidate changed regions at the centre, bottom and lower-right of the image.")[0]
    assert set(map(tuple, c["claim"]["location"]["cells"])) == {(1, 1), (2, 0), (2, 1), (2, 2)}


def test_insufficient_evidence_is_abstention():
    assert triples("The evidence is insufficient to determine changes.") == [("scene", "insufficient_evidence", None)]


def test_contract_shape():
    c = extract_claims("two houses are built at the top .", pair_id="val/x", run_id="r")[0]
    rec = to_contract(c)
    assert rec["decision"]["allowed_outcomes"] == ["accept", "reject", "defer"]
    assert rec["candidate_region_ids"] == [] and rec["claim"]["quantity"] == 2
    assert rec["human_reference"] == "stored_separately_and_never_passed_to_inference"


def test_model_phrasings():
    got = triples("In Image T2, there is a noticeable increase in the number of houses compared to Image T1.")
    assert got == [("building", "added", None)]
    got = triples("There is a visible road that was not present in Image T1.")
    assert got == [("road", "added", 1)]
    got = triples("A small shed visible in T1 is no longer present in T2.")
    assert got == [("building", "removed", 1)]


def test_copula_appears_is_not_addition():
    assert triples("The road in Image T2 appears to be wider.") == [("road", "modified", None)]


def test_landmark_is_not_a_claim():
    got = triples("In T2, there is a new building on the left side of the road.")
    assert got == [("building", "added", 1)]
    got = triples("two rows of villas appear on both sides of the road .")
    assert [o for o, _, _ in got] == ["building"]


def test_negated_object_scope():
    got = triples("No new buildings are present, and no other buildings or road changes are evident.")
    assert {a for _, a, _ in got} == {"no_change"}


def test_change_in_attribute_is_modification():
    assert triples("There is a noticeable change in the road's surface.") == [("road", "modified", None)]
    assert ("building", "modified", None) in triples("There is a noticeable change in the location of the building.")


def test_present_tense_absent_in_t1():
    assert triples("In Image T2, there is a road that is not present in Image T1.") == [("road", "added", 1)]


def test_object_relative_side_is_not_image_location():
    c = extract_claims("There is a new car parked on the right side of the road.")[0]
    assert c["claim"]["location"] is None or c["claim"]["location"].get("relational")
    c = extract_claims("A new building appears on the right side of the image.")[0]
    assert c["claim"]["location"]["cols"] == [2]


def test_v3_same_path_is_not_no_change():
    got = triples("In T2, a paved road has been constructed along the same path, replacing the dirt road.")
    assert ("road", "added", 1) in got and all(a != "no_change" for _, a, _ in got)
    assert triples("The scene remains the same.") == [("scene", "no_change", None)]


def test_v3_yard_and_driveway_are_landmarks():
    got = triples("A tree has appeared in the yard of the house at the bottom center.")
    assert [o for o, _, _ in got] == ["vegetation"]
    got = triples("A dark-colored vehicle appears parked on the driveway of the building, which is not present in T1.")
    assert [o for o, _, _ in got] == ["vehicle"]


def test_v3_vehicle_location_is_not_a_road_claim():
    got = triples("In T2, a red vehicle appears on the road near the center-right of the image.")
    assert [o for o, _, _ in got] == ["vehicle"]
    got = triples("In T2, there is a car parked in front of the building, which is not present in Image T1.")
    assert [o for o, _, _ in got] == ["vehicle"]


def test_v3_now_occupied_by():
    got = triples("The area previously covered in dense vegetation and dirt is now occupied by residential structures and manicured lawns.")
    assert ("building", "added", None) in got and ("vegetation", "removed", None) in got
