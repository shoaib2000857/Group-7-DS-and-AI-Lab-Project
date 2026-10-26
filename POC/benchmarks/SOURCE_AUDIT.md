# Source audit: model access and experiment readiness

Reviewed **26 October 2026**. Only primary author repositories, publisher model cards, official runtime documentation and official library documentation support the plan. No paper leaderboard number is used as a TerraWatch measurement.

## Verification levels

- **Metadata-listed:** a model file is present in the publisher's public file inventory. It has not been downloaded, hashed or loaded locally.
- **Link-identified:** upstream provides a weight link, but payload access/format has not been tested.
- **Runnable-verified:** requires a saved local/Colab successful forward pass. **No candidate has this status yet.**
- **Deferred:** known access, cost, environment or scope reason prevents making it part of the mandatory queue.

`source_metadata.json` records API retrieval status, publisher revisions, weight filenames and relevant Open-CD config paths. Revisions pin source metadata; they do not prove checkpoint fidelity, licence completeness or reproducibility.

## Candidate source register

| Source | What was checked | Status / consequence |
|---|---|---|
| [Open-CD code](https://github.com/likyoo/open-cd), [author weight zoo](https://huggingface.co/likyoo/Open-CD_Model_Zoo) | Supported detector list, installation/test examples, repo config paths and exact HF checkpoint filenames | D01–D08 metadata-listed; matched paths in work package A. MMCV/PyTorch compatibility untested |
| [ChangeMamba author code](https://github.com/ChenHongruixuan/ChangeMamba) | Full task checkpoint links and inference instructions versus VMamba encoder initialisation | D09 link-identified; isolate Linux/CUDA extension setup and source-domain exposure |
| [Chg2Cap](https://github.com/ShizhenChang/Chg2Cap) | Author pretrained-model Google Drive link | C01 link-identified, not payload-tested |
| [RSCaMa code](https://github.com/Chen-Yang-Liu/RSCaMa), [weights](https://huggingface.co/lcybuaa/RSCaMa) | Author checkpoint pointer and model-file inventory | C02 metadata-listed; legacy dependency readiness untested |
| [Change-Agent code](https://github.com/Chen-Yang-Liu/Change-Agent), [weights](https://huggingface.co/lcybuaa/Change-Agent) | Perception checkpoint/inference instructions and weight inventory | C03 metadata-listed; run MCINet without needing the agent/API layer |
| [TEOChat code](https://github.com/ermongroup/TEOChat), [weights](https://huggingface.co/jirvin16/TEOChat) | Official model pointer and weight shards | C08 metadata-listed; pretrained source data may overlap proposed benchmarks |
| [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Author card and weight inventory | Shared C04/W02/J06 checkpoint, not three independent models |
| [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Author card and weight inventory | W01 metadata-listed; missing API licence field must be resolved from actual terms before redistribution |
| [Gemma-3-4B](https://huggingface.co/google/gemma-3-4b-it) | Publisher card, terms/gating and file inventory | C05/W03 metadata-listed but gated; each downloader needs appropriate access |
| [InternVL3.5-4B](https://huggingface.co/OpenGVLab/InternVL3_5-4B) | Author card's separate multi-image example and file inventory | C06/W04/J07; preserve processor's patch grouping and custom-code revision |
| [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | Author interleaved multi-image example and file inventory | C07/W05 metadata-listed |
| [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5) | Author multi-image capability, model class and file inventory | W06 optional larger model; no task-specific superiority claimed |
| [d3-mini publisher](https://huggingface.co/vllm-sr/d3-mini), [Ollama package](https://ollama.com/library/d3) | Native decision source and separate weight/readout inventory; runtime modality/version notes | J01 metadata-listed; verify readout path and installed runner, not just language-model loading |
| [Kev publisher](https://huggingface.co/jaredpalmer/kev-4b), [Ollama listing](https://ollama.com/library/kev) | Text-only task; base/adapter/pointer-head dependency; platform-specific packaging notes | J02 metadata-listed adapter/head, **not complete standalone weights**; prefer publisher Linux runner for Colab |
| [Clef-flash](https://ollama.com/library/clef-flash), [Clef](https://ollama.com/library/clef) | Official platform model descriptions, modality and decision endpoint examples | J03/J04 link-identified. Docs/SDK surfaces vary; a local API feature probe is mandatory. Cloud is not the default |
| [Qwen3 text judge](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Author text-only card and file inventory | J05 metadata-listed; no original-pixel verification capability |
| [Jev release](https://typesafe.ai/blog/jev-1-14) | Official announcement and experimental image support | J08 deferred: actual API access, costs and served version not verified |

## Deliberately excluded from the immediate required queue

- [ChangeChat](https://github.com/hanlinwu/ChangeChat): official pretrained weights section says coming soon. Literature relevance is not evidence of current inference readiness.
- [CCExpert](https://github.com/Meize0729/CCExpert): author model zoo has Baidu-hosted 0.5B/7B links; legitimate payload access not tested. Reserve rather than blocking specialist work.
- Additional giant VLMs, foundation segmentation systems and semantic transition heads: useful possible follow-ups, but would add new label/setup requirements before the present comparison is stable.

## Infrastructure / preprocessing sources

- [Official Colab FAQ](https://research.google.com/colaboratory/faq.html): resources vary; no particular GPU/concurrency assumption follows from Pro+.
- [OpenCV tracking/ECC API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html), [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html), [CLAHE tutorial](https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html).
- [scikit-image exposure API](https://scikit-image.org/docs/stable/api/skimage.exposure.html): histogram/intensity operations. Their existence does not demonstrate that they improve satellite change detection.

## What remains unverified

No weights/datasets downloaded; no Colab/WSL bridge exercised; no GPU job launched; no remote service paid for; no benchmark score measured. Actual checkpoint licences can differ from code licences. No immutable runtime digest is yet pinned for Ollama-hosted packages; record it at download time. The full-model checkpoint provenance and all LEVIR-family cross-release overlaps remain audit tasks, not settled facts.
