# B. Sujal Raj Pradhan: descriptions and claim extraction

Own image-pair caption generation, reference-caption evaluation and conversion into atomic claims. Models should see the same evidence condition; do not compare one model with oracle masks against another with raw images.

## Eight candidate models

| ID | Model / exact identity | Purpose | Priority / access and setup |
|---|---|---|---|
| C01 | [Chg2Cap](https://github.com/ShizhenChang/Chg2Cap) | Conventional specialist caption baseline | P2; author Google Drive weight link identified; payload not tested; S/M |
| C02 | [RSCaMa](https://github.com/Chen-Yang-Liu/RSCaMa) | State-space change captioning baseline | P1; `lcybuaa/RSCaMa`, file `RSCaMa.pth`; isolate Mamba/legacy dependencies; S/M |
| C03 | [MCINet from Change-Agent](https://github.com/Chen-Yang-Liu/Change-Agent) | Closest integrated detection/caption precedent | P2; `lcybuaa/Change-Agent`, file `MCI_model.pth`; run perception only, not a paid agent service; S/M |
| C04 | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | General multi-image VLM | P1; author weights listed; M |
| C05 | [Gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | Alternative visual-language family | P2; gated publisher access/terms; M |
| C06 | [InternVL3_5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) | Dynamic-tile multi-image alternative | P2; author weights and multi-image example; M |
| C07 | [SmolVLM2-2.2B-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | Small model/resource control | P1; author weights and interleaved-image example; S/M |
| C08 | [TEOChat](https://github.com/ermongroup/TEOChat) | Domain-specific temporal EO assistant | P3; `jirvin16/TEOChat` weight shards listed; legacy multimodal stack; M/L |

The model identities reused by C and D are the **same downloaded checkpoint**, with distinct role prompts and outputs. Coordinate caches with Priyanshu/Shoaib. No stage result constitutes independent agreement just because the role changed.

## Source findings affecting the queue

- **ChangeChat is deferred**, not an immediate ninth checkpoint: its [official README](https://github.com/hanlinwu/ChangeChat#pre-trained-weights) still says weights are coming soon. Do not replace it with arbitrary community weights without provenance.
- **CCExpert-0.5B/7B** is a reserve specialist: the [author model zoo](https://github.com/Meize0729/CCExpert#model-zoo) links Baidu-hosted weights. Access is unverified, and the environment is separate. Swap it in only after obtaining a legitimate checkpoint; it is not needed to satisfy the eight-model pool.
- Specialist tokenizers/vocabularies, reference JSON and image normalisation are part of the checkpoint contract. Do not assume `.pth` means a Transformers model.

## Run conditions

**B0, mandatory:** image pair only, fixed instruction and output-length limit. Use the official specialist inference path; do not claim it takes instructions if it does not.

**B1, diagnostic:** images plus predicted region evidence for instruction-capable models. A specialist that lacks such an interface is marked `unsupported_condition`, not scored as if it ignored evidence. Tuning a new specialist evidence input is later research work.

**B2, diagnostic only:** oracle masks, clearly segregated. Never pass oracle labels into the deployable chain.

Start with this instruction, then freeze any revision on development data:

> Image T1 is earlier and T2 is later. Describe visible building or road changes in at most two sentences. Use image-relative locations. If the evidence is insufficient, state that. Do not infer legality, ownership or cause. Do not give an exact count unless individual objects are distinguishable. If no relevant change is visible, say so.

Do not force native specialist outputs into the same length by truncating mid-sentence. Record their decoder limits and show both the natural output and any standardised evaluation representation. Keep native caption metrics supplementary.

## Deliverables

1. RSCaMa and one general VLM adapter first; SmolVLM2 is a useful fast fallback. Save raw model output and request parameters before parsing.
2. A deterministic/manual-audited extraction baseline with fields `object`, `action`, `location`, `quantity`, `source_span`, `evidence_region_ids`. Avoid adding a large parser benchmark before caption inference works. If a parser model is used, record its identity and evaluate its errors separately.
3. An evidence-template control that says only what the input encodes, e.g. a candidate changed region at an image-relative location. It is a reporting control, not a learned captioner.
4. Per-model records for BLEU-4/METEOR/ROUGE-L/CIDEr using one pinned script and all available references; Java/tokenisation requirements belong in the environment notes. Also collect omission, false-change and human factual judgments.
5. Cached captions/claims from a frozen baseline and one challenger for Shoaib. Include unchanged cases; do not select only visually impressive examples.

Human-audited precision/coverage and visual failure examples determine the shortlist together with runtime. A pretrained captioner may have trained on LEVIR images later used here: disclose that provenance, and never call this a clean zero-shot geographic test without evidence. No full captioner training is planned during this initial screen; one small adapter experiment can follow only after selecting a baseline.
