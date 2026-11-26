"""WSL orchestrator for Colab L4 #2 (m5-cap-ver): WS1-Cap (1,858 Test pairs + BERTScore/CIDEr/METEOR + 200-pair probe) + WS2 (150-pair blind Test Hybrid Verifier + 10 E2E traces + 120-claim blind audit packets)."""
from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

COLAB = "/home/shoaib/.local/bin/colab"
REPO = Path("/mnt/d/IITM/DegreeTerm3/DSAI/Project")
M4 = REPO / "POC/milestone4/rework"
M5 = REPO / "POC/milestone5"
TMP_D = M5 / "tmp_cap_ver"
SESSION = "m5-cap-ver"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def exec_bash(session: str, bash_cmd: str, tag: str, timeout: int = 7200) -> None:
    TMP_D.mkdir(parents=True, exist_ok=True)
    runner = TMP_D / f"_run_{tag}.py"
    runner.write_text(
        "import subprocess, sys\n"
        f"cmd = {bash_cmd!r}\n"
        "p = subprocess.Popen(['bash', '-lc', cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n"
        "for line in p.stdout:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
        "rc = p.wait()\n"
        "if rc != 0:\n"
        "    raise RuntimeError(f'Command failed with rc={rc}')\n"
        f"print('___OK_{tag}___', flush=True)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [COLAB, "exec", "--session", session, "--file", str(runner), "--timeout", str(timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0 or f"___OK_{tag}___" not in proc.stdout:
        raise RuntimeError(f"Colab exec {tag} failed (rc={proc.returncode})")


def main() -> None:
    TMP_D.mkdir(parents=True, exist_ok=True)
    code_tar = TMP_D / "m5_cap_code.tar"
    if code_tar.exists():
        code_tar.unlink()

    print("=== Packaging m5-cap-ver code + manifests + detector checkpoint ===", flush=True)
    with tarfile.open(code_tar, "w") as tf:
        tf.add(M5 / "scripts/m5_eval_captioners_and_verifier.py", arcname="scripts/m5_eval_captioners_and_verifier.py")
        for fn in ("colab_setup_captioner.py", "detector_models.py", "pipeline_recovery.py", "train_captioner.py"):
            tf.add(M4 / "scripts" / fn, arcname=f"scripts/{fn}")
        for p in sorted((REPO / "src").rglob("*.py")):
            if "__pycache__" not in p.parts:
                tf.add(p, arcname=f"src/{p.relative_to(REPO / 'src').as_posix()}")
        for p in sorted((M4 / "evidence/portable_data").glob("*")):
            if p.is_file():
                tf.add(p, arcname=f"data/portable_data/{p.name}")
        tf.add(M4 / "runs/detector/D03_bit_lr3e4/best.pt", arcname="checkpoints/D03_bit_lr3e4/best.pt")
        tf.add(M4 / "evidence/captioner_runs/C06_rscama_frozen_clip/eval_test_captions.jsonl", arcname="inputs/c06_test_captions.jsonl")
        tf.add(M4 / "evidence/captioner_runs/C00_rscama_released/eval_test_captions.jsonl", arcname="inputs/c00_test_captions.jsonl")
        tf.add(M4 / "evidence/verification/verification_claims.jsonl", arcname="inputs/val_claims.jsonl")
    print(f"Code bundle ready: {code_tar} ({code_tar.stat().st_size/(1024*1024):.1f} MB)", flush=True)

    started = False
    try:
        for gpu in ("L4", "T4"):
            r = run([COLAB, "new", "--session", SESSION, "--gpu", gpu], check=False)
            if r.returncode == 0:
                started = True
                break
        assert started, "Could not acquire L4 or T4 GPU session"
        run([COLAB, "upload", "--session", SESSION, str(code_tar), "/content/m5_cap_code.tar"])
        code_tar.unlink(missing_ok=True)

        # Upload captioner checkpoints directly (uncompressed, ~2 min each)
        exec_bash(SESSION, "mkdir -p /content/terrawatch/checkpoints/C06 /content/terrawatch/checkpoints/C01 /content/terrawatch/checkpoints/C04 && tar -xf /content/m5_cap_code.tar -C /content/terrawatch && rm -f /content/m5_cap_code.tar", tag="unpack_code", timeout=300)
        for tag_c, run_id in (("C06", "C06_rscama_frozen_clip"), ("C01", "C01_rscama_ref"), ("C04", "C04_rscama_adamw_cos")):
            src = M4 / "runs/captioner" / run_id / "best.pth"
            assert src.is_file(), f"Missing {src}"
            run([COLAB, "upload", "--session", SESSION, str(src), f"/content/terrawatch/checkpoints/{tag_c}/best.pth"])

        # Bootstrap LEVIR-MCI, RSCaMa, CLIP, BIT_CD, ChangeFormer, bert-score, qwen-vl-utils
        exec_bash(
            SESSION,
            "set -e; BASE=/content/terrawatch; "
            "python3 -m pip install -q einops timm ftfy regex nltk scipy bert-score qwen-vl-utils accelerate; "
            "if [ ! -d $BASE/models/external/BIT_CD ]; then git clone --filter=blob:none --no-checkout https://github.com/justchenhao/BIT_CD.git $BASE/models/external/BIT_CD && git -C $BASE/models/external/BIT_CD checkout --detach adcd7aea6f234586ffffdd4e9959404f96271711; fi; "
            "if [ ! -d $BASE/models/external/ChangeFormer ]; then git clone --depth 1 https://github.com/wgcban/ChangeFormer.git $BASE/models/external/ChangeFormer; fi; "
            "python3 $BASE/scripts/colab_setup_captioner.py",
            tag="setup_cap_ver",
            timeout=1800,
        )

        exec_bash(
            SESSION,
            "set -e; BASE=/content/terrawatch; "
            "python3 $BASE/scripts/m5_eval_captioners_and_verifier.py "
            "--manifests-dir $BASE/data/portable_data "
            "--extracted-root $BASE/data/extracted "
            "--detector-ckpt $BASE/checkpoints/D03_bit_lr3e4/best.pt "
            "--bit-source-dir $BASE/models/external/BIT_CD "
            "--changeformer-source-dir $BASE/models/external/ChangeFormer "
            "--rscama-source-dir $BASE/models/external/RSCaMa "
            "--c06-test-captions $BASE/inputs/c06_test_captions.jsonl "
            "--c00-test-captions $BASE/inputs/c00_test_captions.jsonl "
            "--c06-ckpt $BASE/checkpoints/C06/best.pth "
            "--c01-ckpt $BASE/checkpoints/C01/best.pth "
            "--c04-ckpt $BASE/checkpoints/C04/best.pth "
            "--val-claims-jsonl $BASE/inputs/val_claims.jsonl "
            "--output-dir $BASE/m5_cap_ver_out; "
            "tar -czf /content/m5_cap_ver_out.tar.gz -C $BASE/m5_cap_ver_out . && ls -lh /content/m5_cap_ver_out.tar.gz",
            tag="run_cap_ver",
            timeout=7200,
        )

        out_tar = TMP_D / "m5_cap_ver_out.tar.gz"
        if out_tar.exists():
            out_tar.unlink()
        run([COLAB, "download", "--session", SESSION, "/content/m5_cap_ver_out.tar.gz", str(out_tar)])

        dst = M5 / "evidence/test_captioner_and_verifier"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_tar, "r:gz") as tf:
            tf.extractall(dst)
        out_tar.unlink(missing_ok=True)
        print(f"=== M5-CAP-VER COMPLETE -> {dst} ===", flush=True)
    finally:
        if started:
            run([COLAB, "stop", "--session", SESSION], check=False)
        shutil.rmtree(TMP_D, ignore_errors=True)


if __name__ == "__main__":
    main()
