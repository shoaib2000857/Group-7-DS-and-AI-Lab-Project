"""Build colab/rscama_finetune.ipynb with the tested sources embedded (%%writefile cells).

  python scripts/make_colab_notebook.py
"""
import json
from pathlib import Path

EXEC_ROOT = Path(__file__).resolve().parents[1]
EMBED = ["scripts/remote_zip.py", "scripts/common.py", "scripts/caption_rscama.py", "colab/rscama_colab.py"]

INTRO = """# TerraWatch B: RSCaMa retrained under the B protocol (Colab or Kaggle GPU)

**What this does.** Retrains RSCaMa from the public CLIP ViT-B/32 weights with the upstream recipe on a seeded 20% of the LEVIR-MCI train split, picks the checkpoint on our 50 development pairs, then captions the **same 50 development + 100 screening pairs** used for every B0 system. The screening pairs are never loaded before that final pass, so they are genuinely unseen (unlike the released checkpoint, which was selected on val). It also runs the released checkpoint on CUDA for the parity check.

**Data.**
- LEVIR-MCI, Hugging Face copy by the authors: https://huggingface.co/datasets/lcybuaa/LEVIR-MCI (revision `dff09c7`)
- Dataset project page: https://github.com/Chen-Yang-Liu/LEVIR-MCI · captions from LEVIR-CC: https://github.com/Chen-Yang-Liu/RSICC
- Used here: 1,313 train pairs for training + 50 train pairs for checkpoint selection (1,363 = 20.0% of 6,815 train) + the 100 val screening pairs. Test split is not downloaded. Only these pairs are fetched (about 400 MB of the 2.8 GB archive, by HTTP range requests).
- Academic use only: keep the runtime and any copies private.

**Model.** RSCaMa, https://github.com/Chen-Yang-Liu/RSCaMa (commit `aa68e4c`); released checkpoint https://huggingface.co/lcybuaa/RSCaMa for the parity run only.

**How to run.** Needs an NVIDIA GPU (a TPU will not work). Training takes roughly 1-2 hours.
- *Colab:* Runtime > Change runtime type > T4 GPU, then Runtime > Run all. Drive is mounted so training resumes after a disconnect: run all again.
- *Kaggle:* Create > New Notebook > File > Import Notebook (this file). In the right sidebar: Session options > Accelerator > GPU T4 x2, and Internet on (needs a phone-verified account). Then Run All. Kaggle sessions stop after about 12 hours idle or total; results are in `/kaggle/working`.

The last cell writes `b_rscama_results.zip` (captions and logs, no images or weights); send that file back.
"""

SETUP = [
    ("code", "!nvidia-smi --query-gpu=name,memory.total --format=csv"),
    ("code", """import os
ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.isdir('/kaggle')
ON_COLAB = not ON_KAGGLE and os.path.exists('/var/colab/hostname')  # Kaggle images also carry Colab env vars
for d in ('/content/b', '/content/cache'):  # same paths on every platform
    os.makedirs(d, exist_ok=True)
import urllib.request
try:
    urllib.request.urlopen('https://huggingface.co', timeout=15)
except Exception as e:
    raise SystemExit('No internet in this session. Kaggle: right sidebar > Session options > Internet on '
                     '(needs a phone-verified account), then run all again. ' + repr(e))
OUT = '/kaggle/working/out' if ON_KAGGLE else '/content/out'
if ON_COLAB:  # Drive keeps last.pth across disconnects; just run all again to resume
    from google.colab import drive
    drive.mount('/content/drive')
    OUT = '/content/drive/MyDrive/terrawatch_b_rscama'
os.makedirs(OUT, exist_ok=True)
os.environ['OUT'] = OUT
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
print('colab' if ON_COLAB else 'kaggle' if ON_KAGGLE else 'other', OUT)"""),
    ("code", """%%bash
set -e
mkdir -p /content/cache /content/b
pip -q install transformers==4.39.3 einops==0.6.1 ftfy regex nltk "setuptools<70" wheel
pip -q install --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
[ -d /content/cache/rscama_src ] || git clone -q https://github.com/Chen-Yang-Liu/RSCaMa /content/cache/rscama_src
git -C /content/cache/rscama_src checkout -q aa68e4c6933c7dd0f2e2f9cbd7642d61929c86a5
git -C /content/cache/rscama_src rev-parse HEAD"""),
]

RUN = [
    ("md", "## 1. Fetch only the selected pairs (hashes are checked against the frozen manifest)"),
    ("code", "!cd /content && python b/rscama_colab.py --data /content/levir --out {OUT} fetch\nassert _exit_code == 0, 'step failed: see the traceback above'"),
    ("md", "## 2. Parity: released checkpoint on CUDA with the same pure-PyTorch shim used on the Mac"),
    ("code", """from huggingface_hub import hf_hub_download
ck = hf_hub_download('lcybuaa/RSCaMa', 'RSCaMa.pth', revision='3e527e6656b62cc615eab1d2f3fb73777cbbf62c',
                     local_dir='/content/weights', token=False)
os.environ['CKPT'] = ck
!cd /content && python b/rscama_colab.py --data /content/levir --out {OUT} parity --ckpt {ck}
assert _exit_code == 0, 'step failed: see the traceback above'"""),
    ("md", "## 3. Train (resumes from `last.pth` if present)\nEach step takes 12 image pairs and decodes all 5 captions of each against one encoding (60 captions per update, upstream uses 64). Prints progress every 25 steps; stops after 8 h so captioning fits in Kaggle's 12 h limit."),
    ("code", "!cd /content && python b/rscama_colab.py --data /content/levir --out {OUT} train --epochs 40 --patience 8 --batch 12 --max-hours 8\nassert _exit_code == 0, 'step failed: see the traceback above'"),
    ("md", "## 4. Caption the development + screening pairs with the best checkpoint"),
    ("code", "!cd /content && python b/rscama_colab.py --data /content/levir --out {OUT} caption\nassert _exit_code == 0, 'step failed: see the traceback above'"),
    ("md", "## 5. Optional: parity with the real Mamba CUDA kernels\nBuilding the kernels can take 10-30 minutes; skip this cell if it fails, the results above are complete."),
    ("code", """%%bash
pip -q install --no-build-isolation causal-conv1d mamba-ssm && \\
cd /content && python b/rscama_colab.py --data /content/levir --out $OUT parity --ckpt $CKPT --kernels \\
|| echo "kernel parity skipped" """),
    ("md", "## 6. Package results (captions + logs only)"),
    ("code", """import glob, hashlib, json, platform, shutil, torch, transformers, zipfile
env = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
       'transformers': transformers.__version__, 'python': platform.python_version(),
       'best_pth_sha256': hashlib.sha256(open(f'{OUT}/best.pth', 'rb').read()).hexdigest()}
json.dump(env, open(f'{OUT}/colab_env.json', 'w'), indent=1)
shutil.copy('/content/levir/fetch.json', f'{OUT}/fetch.json')
ZIP = '/kaggle/working/b_rscama_results.zip' if ON_KAGGLE else '/content/b_rscama_results.zip'
with zipfile.ZipFile(ZIP, 'w') as z:
    for p in glob.glob(f'{OUT}/*.json') + glob.glob(f'{OUT}/*.jsonl'):
        z.write(p, os.path.basename(p))
print(env, ZIP)
if ON_COLAB:
    from google.colab import files
    files.download(ZIP)
else:
    print('Download', ZIP, 'from the Output panel (Kaggle: right-hand sidebar > Output).')"""),
]


def cell(kind, src):
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


def main():
    cells = [cell("md", INTRO)] + [cell(k, s) for k, s in SETUP]
    cells.append(cell("md", "## Sources (same files as `POC/execution`, tested locally)"))
    for rel in EMBED:
        name = Path(rel).name
        cells.append(cell("code", f"%%writefile /content/b/{name}\n" + (EXEC_ROOT / rel).read_text()))
    cells += [cell(k, s) for k, s in RUN]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = EXEC_ROOT / "colab" / "rscama_finetune.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    print(out, len(cells), "cells")


if __name__ == "__main__":
    main()
