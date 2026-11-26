## Appendix F. Sample Predictions (10 Authentic Held-Out Test End-to-End Traces)

Ten authentic held-out LEVIR-MCI Test image pairs spanning all operational outcomes (accepted building, accepted road, accepted unchanged, rejected missed-change hallucination, out-of-ontology deferral, exact-count deferral, selective VLM escalation, small change, and mixed multi-claim scene) were exported as full Evidence Review UI bundles under `POC/milestone5/evidence/test_captioner_and_verifier/test_e2e_traces/`. Visual panels for five representative cases appear in Section 6.3 (`figures/m5_test_e2e_traces.png`).

| Trace Category | Held-Out Test Pair ID | Generated Caption (`C06`) | Claim Outcomes (`M5 Hybrid`) | Operational Behavior Demonstrated |
|---|---|---|---|---|
| `test_accepted_building` | `levir_mci/test/test_000032` | *"some buildings appear in the woods"* | `['accept', 'defer']` | Building claim grounded & accepted via BIT mask; *"woods"* mention deferred out-of-ontology. |
| `test_accepted_road` | `levir_mci/test/test_000789` | *"a road appears in the bareland"* | `['accept']` | Continuous road construction cleanly segmented and mask-first accepted. |
| `test_accepted_unchanged` | `levir_mci/test/test_000013` | *"the two scenes seem identical"* | `['accept']` | Zero predicted change pixels confirm unchanged scene without false alarm. |
| `test_rejected_hallucination` | `levir_mci/test/test_000175` | *"the scene is the same as before"* | `['reject']` | `C06` missed a new cul-de-sac house; `D03` detected it and verifier rejected the false `no_change`. |
| `test_rejected_second` | `levir_mci/test/test_001038` | *"the two scenes seem identical"* | `['reject']` | Hedged `no_change` escalated to Qwen3-VL-4B and flagged for analyst check. |
| `test_deferred_out_of_ontology` | `levir_mci/test/test_000381` | *"a villa is built in the woods"* | `['accept', 'defer']` | Villa accepted via building mask; out-of-ontology *"woods"* deterministically deferred. |
| `test_deferred_exact_count` | `levir_mci/test/test_000389` | *"two houses are built in the woods"* | `['defer', 'defer']` | Exact numeric count (*"two"*) deterministically deferred without instance-counting ground truth. |
| `test_vlm_escalated_case` | `levir_mci/test/test_000695` | *"the two scenes seem identical"* | `['accept']` | Hedged `no_change` escalated to Direct Qwen3-VL-4B and confirmed (`accept`). |
| `test_small_change_case` | `levir_mci/test/test_000050` | *"a house appears in the top left corner of the scene"* | `['accept']` | Small localized building change (<1% area) localized to top-left 3×3 grid cell and accepted. |
| `test_mixed_scene_case` | `levir_mci/test/test_000123` | *"a road is built across the woods and some houses are built along the road"* | `['accept', 'accept']` | Multi-claim compound caption decomposed into road + building atomic claims, both grounded and accepted. |
