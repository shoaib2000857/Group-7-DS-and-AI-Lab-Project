## Appendix G. Failure Case Examples

Representative held-out Test failure cases mined automatically across the detector (`detector_error_mining.json`), captioner (`captioner_test_full.json`), and hybrid verifier (`test_verification_claims.jsonl`) are catalogued below (visual error maps in Section 10.2, `figures/m5_error_taxonomy_gallery.png`).

### G.1 Detector Failure Examples (`D03_bit_lr3e4`, 1,858 Held-Out Test Pairs)

| Failure Mode | Pair ID | GT Changed Pixels | Pred Changed Pixels | Binary F1 (%) | Observed Physical Root Cause |
|---|---|---:|---:|---:|---|
| Unchanged FP (E1) | `levir_mci/test/test_000977` | 0 | 2,593 (bldg) | 0.0% | Deep seasonal cast shadows across identical suburban roofs. |
| Unchanged FP (E1) | `levir_mci/test/test_002083` | 0 | 1,648 (bldg+road) | 0.0% | Winter canopy thinning exposing existing driveway and roofline. |
| Large FN (E3) | `levir_mci/test/test_001303` | 14,009 (11,259 bldg, 2,750 road) | 0 | 0.0% | Bright specular warehouse roof glare saturating $T_1$ contrast. |
| Large FN (E3) | `levir_mci/test/test_001393` | 5,840 | 0 | 0.0% | Low-contrast asphalt road widening without strong edge shift. |
| Road/Bldg Swap (E4) | `levir_mci/test/test_001217` | 20,396 | 18,532 (3,492 swap) | 82.5% | Flat grey industrial roof adjacent to highway misclassified as road. |
| Tiny Miss (E2) | `levir_mci/test/test_000226` | 141 (bldg) | 0 | 0.0% | Isolated 141-pixel shed occluded by surrounding dark tree canopy. |

### G.2 Captioner & Hybrid Verifier Failure Examples

| Module & Mode | Pair ID | Caption / Claim Span | Verifier Decision vs Proxy GT | Root Cause |
|---|---|---|---|---|
| Captioner Hallucination (E5) | `levir_mci/test/test_000010` | *"a small house appears in the bareland"* (GT: 0 px) | Caught & rejected by `D03` mask | `C06` triggers false construction phrase on bare soil texture shift. |
| Captioner Obj Swap (E5) | `levir_mci/test/test_000064` | *"a road is built on both sides of the road"* (GT: 5,723 bldg, 0 road) | Caught by class-specific mask check | Repetitive decoder n-gram loop confusing linear building rows with a road. |
| Captioner Large Miss (E5) | `levir_mci/test/test_001307` | *"the two scenes seem identical"* (GT: 22,812 bldg, 3,320 road) | Rejected by `D03` mask when detected | Conservative no-change prior in dense urban grid redevelopment. |
| Verifier False Accept (E6) | `levir_mci/test/test_001422` | *"a road appears on the bareland"* | `accept` (`mask_first_accept`) vs `contradicted` | Upstream `D03` road false positive propagated into mask-first accept. |
| Verifier VLM Ambiguity (E6) | `levir_mci/test/test_001729` | *"the two scenes seem identical"* | `accept` (`vlm_escalate`) vs `contradicted` | Escalated Qwen3-VL-4B missed small subtle building edit on full-tile view. |
| Verifier False Reject (E6) | `levir_mci/test/test_001038` | *"the two scenes seem identical"* | `reject` (`vlm_escalate`) vs `supported` | Seasonal color shift caused escalated VLM to over-flag change on unchanged tile. |
