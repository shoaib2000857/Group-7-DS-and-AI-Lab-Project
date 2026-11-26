## 14. Possible Improvements

Each proposal below addresses a weakness measured in Milestone 4 or Milestone 5 and states the metric against which it would be judged. Proposals are ordered by the severity of the weakness they target, not by implementation cost.

**Table 14.1. Weakness-to-remedy map.**

| Observed weakness | Evidence | Proposed remedy | Success criterion |
|---|---|---|---|
| Out-of-ontology change | SECOND-CC Test any-change F1 23.17%, recall 14.46%, FA@50 25.08% | Multi-class, multi-sensor pretraining and fine-tuning | SECOND-CC any-change F1 and unchanged FA@50 versus the current values |
| Small-change recall | Test small-change F1 48.87% against binary F1 86.83% | Tiling with overlap, small-object loss weighting, higher-resolution decoder | Small-change F1 with no rise in FA@50 |
| Blur and compression fragility | D03 38.22% mIoU at blur radius 2.0; 67.99% at JPEG q20 | Degradation-aware augmentation, ensembling with a sharp-feature model | Corruption-ladder mIoU at the failing conditions |
| Proxy-label verification | 120 pairs, 147 claims, labels from masks and reference captions | Human-labelled, larger and test-split verification audit | Human-verified groundedness with Cohen's kappa |
| Referee cost and size | 1.225 referee calls per pair under direct policy; 4B bf16 model | Quantised or distilled referee; selective routing | Cost per 1,000 pairs (Table 12.7) at equal precision |
| Template-like captions | 388-word vocabulary; changed-pair BLEU-4 0.3681 against 0.6128 corpus | Larger vocabulary, tokenizer pretraining, instruction-tuned decoder | Changed-pair BLEU-4 and missed-change rate (11.78%) |
| Seed sensitivity | C06 Val BLEU-4 spread 0.6297 +/- 0.0183 | More seeds, checkpoint averaging | Narrower spread |
| Annotation scarcity | Review corrections are not yet used for learning | Active learning from the Evidence Review audit store | Accuracy gain per labelled correction |

### 14.1 Better Datasets and More Training Data

The largest measured gap is ontological: LEVIR-MCI supplies only road and building change from one region and sensor, and zero-shot transfer reaches 23.17% any-change F1. Joint training on multi-class, multi-sensor change corpora, with SECOND-CC (3,241 training pairs already catalogued in the portable receipt) as the immediate candidate and further sources listed in `configs/datasets.yaml` (for example S2Looking, SYSU-CD, LEVIR-CD+), would widen the class set and sensor coverage. Mixed-dataset training must keep the governed split discipline: manifests with pair-identifier digests, a SECOND-CC test split held out entirely, and no reference caption or mask reaching inference. Per-tile dates and coordinates, which the current record schema lacks, should be collected for any new corpus so that seasonal and geographic bias can be measured per stratum. Scene-level class balance deserves attention because road pixels are only 3.4% of Test pixels (4.12 of 121.77 million); oversampling road-bearing tiles is a direct response to the lower road IoU (74.53% against 77.33% for buildings on Test).

### 14.2 Improved Preprocessing

Registration and radiometric mismatch are the dominant sensitivity of the detector (63.88% mIoU at a 4-pixel shift). Automatic sub-pixel co-registration before inference, histogram matching between T1 and T2, and shadow-invariant colour transforms would address them at the input and reuse the alignment-jitter diagnostic that the OpenCV witness already computes. For scenes larger than 256 x 256, sliding-window tiling with overlap and probability blending removes dependence on tile boundaries and gives the detector context that a single tile lacks, which targets the small-change recall gap. Photometric augmentation applied independently to T1 and T2 lowered clean mIoU in the ablation (D15 68.34%, D16 68.55% against 71.82% for D01), so radiometric augmentation should be applied consistently to both dates or replaced by explicit normalisation.

### 14.3 Alternative Models and Better Feature Engineering

Three architecture directions follow from the results. First, the Siam-ResNet18-UNet baseline wins on clean data (76.73% against 75.93% Test mIoU, 8.23% FA@50) and BIT wins under blur and transfer; a feature-level fusion that gives the UNet decoder BIT's token context targets both. Second, ChangeFormer (41.04 million parameters) underperformed (70.46% Val mIoU) under the 25-epoch budget, so pretrained remote-sensing backbones or longer schedules are the fair test of whether larger transformers help here. Third, replacing the template-like captioner head with a decoder pretrained on a larger caption vocabulary addresses the changed-pair BLEU-4 deficit. Feature engineering that targets small changes includes a high-resolution side branch, edge-aware losses and explicit class-specific heads for road and building extent, since 15.81% of road pixels and 12.57% of building pixels are lost to background. For the referee, the 4-view result (72.41% precision against 82.43%) indicates that crop prompting needs calibration, so prompt design and few-shot examples drawn from the Evidence Review corrections are candidate improvements.

### 14.4 Hyperparameter Tuning

Milestone 4 varied one factor at a time on a single seed. The decisive gains came from head learning rate (D02 64.55%, D01 71.82%, D03 74.54%) and backbone unfreezing (D13 62.45% frozen), which interact; a joint search over head and backbone learning rates, freeze duration and schedule, run with at least three seeds, is the principled next step. The captioner's seed spread exceeds several arm gaps, so tuning there requires multi-seed evaluation before ranking. The verifier operating point (tau = 0.70) and routing triggers were tuned on a single Validation audit; they should be re-estimated on a larger human-labelled set with calibration error (ECE) and selective-risk curves as the objective.

### 14.5 Ensemble Methods

Averaging the logits of D03 and A03 combines complementary behaviour: A03 leads on clean mIoU by 0.80 points and D03 leads by 12.38 and 30.36 points under blur radius 1.0 and 2.0. A simple logit average or a sharpness-gated mixture, evaluated on the full 14-condition stress matrix and on SECOND-CC, tests whether the gain exceeds the two-model inference cost measured in Table 12.4 (combined P50 latency 23.58 ms on the Colab Tesla T4: D03 16.93 ms plus A03 6.65 ms). Seed ensembling of the three D03 and three C06 runs already available offers a lower-cost variance reduction. For verification, disagreement between the mask checker and the witness is already used as a routing trigger (51 trigger events of type `T_witness_disagrees_no_visual_change` across the 95 triggered claims); ensembling several witnesses is a natural extension.

### 14.6 Efficiency and Deployment

The referee dominates cost (Section 12.6). Distilling its accept, reject and defer behaviour into a smaller model, or quantising it to 4-bit weights, reduces memory and per-call latency; both must be validated against the hybrid verifier's precision and deferral rates before adoption. The mask-only policy already attains 91.38% proxy precision with zero referee calls, so raising its coverage through better detector calibration shrinks the referee workload directly. The Triton selective-scan kernel is CUDA-specific; an MPS or CPU-friendly scan would remove the dependency for edge review stations.

### 14.7 Future Research Directions

Human-in-the-loop learning is the most promising direction: the Evidence Review application already records overrides in an append-only audit store, and these corrections are labelled hard cases targeted at exactly the claims the system gets wrong. Active selection of uncertain claims for review, followed by periodic fine-tuning with held-out tests, closes the loop between deployment and training. Further directions include temporal sequences beyond two dates, uncertainty-aware captioning that abstains on small or occluded changes, quantity estimation from masks for numeric claims that are currently deferred, and a fairness audit across regions and sensors once per-tile metadata exists.
