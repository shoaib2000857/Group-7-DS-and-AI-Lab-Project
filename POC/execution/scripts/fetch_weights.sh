#!/bin/sh
# Pinned, token-free downloads of public weights in priority order (plain HTTPS, resumable).
export HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_ENABLE_HF_TRANSFER=0
PY=.venv-vlm/bin/python
$PY - <<'PYEOF'
from huggingface_hub import snapshot_download, hf_hub_download
print(snapshot_download('HuggingFaceTB/SmolVLM2-2.2B-Instruct', revision='482adb537c021c86670beed01cd58990d01e72e4', token=False), flush=True)
print('SMOL_DONE', flush=True)
print(snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17', token=False), flush=True)
print('QWEN_DONE', flush=True)
print(hf_hub_download('lcybuaa/RSCaMa', 'RSCaMa.pth', revision='3e527e6656b62cc615eab1d2f3fb73777cbbf62c', local_dir='weights/rscama', token=False), flush=True)
print('RSCAMA_DONE', flush=True)
PYEOF
echo ALL_DONE
