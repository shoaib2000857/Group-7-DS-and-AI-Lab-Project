"""Unified WSL orchestrator for Colab GPU (m5-all):
Runs BOTH Milestone 5 evaluation suites on ONE bootstrapped VM (zero duplicate LEVIR-MCI downloads!):
  Part A: m5_eval_detectors_and_stress.py (5 detectors + controls on 1,858 Test pairs, ROC/PR, 3x3 CM,
          5-bucket stratification, bootstrap CIs & Wilcoxon/McNemar, 17-condition stress suite, WS4-Det mining)
  Part B: m5_eval_captioners_and_verifier.py (C06/C00/C01/C04/Template/Always-NC on 1,858 Test pairs with
          BLEU-1..4/ROUGE-L/METEOR/CIDEr/BERTScore, 200-pair stress probe, WS2 150-pair blind Test Hybrid Verifier,
          10 Test E2E traces, and 120-claim blind human-audit packet + visual strips)

Uses chunked 250 MB uploads for 1.67 GB captioner checkpoints (bypassing Jupyter's single-POST size limit)
and verifies SHA-256 on Colab before execution. Always stops the Colab session in finally:.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

COLAB = "/home/shoaib/.local/bin/colab"
REPO = Path("/mnt/d/IITM/DegreeTerm3/DSAI/Project")
M4 = REPO / "POC/milestone4/rework"
M5 = REPO / "POC/milestone5"
TMP_D = M5 / "tmp_m5_all"
SESSION = "m5-all"
CHUNK_BYTES = 45 * 1024 * 1024  # 45 MB per chunk (strictly below Jupyter 100 MB single-POST limit)


def run(cmd: list[str], check: bool = True, retries: int = 1) -> subprocess.CompletedProcess:
    last = None
    for attempt in range(1, retries + 1):
        print(f"+ [try {attempt}/{retries}] " + " ".join(cmd), flush=True)
        last = subprocess.run(cmd, check=False)
        if last.returncode == 0:
            return last
        if attempt < retries:
            time.sleep(5 * attempt)
    if check and last is not None:
        raise subprocess.CalledProcessError(last.returncode, cmd)
    return last


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
        f"print('___OK_' + {tag!r} + '___', flush=True)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [COLAB, "exec", "--session", session, "--file", str(runner), "--timeout", str(timeout)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode != 0 or f"___OK_{tag}___" not in proc.stdout:
        raise RuntimeError(f"Colab exec {tag} failed (rc={proc.returncode})")


def upload_chunked(session: str, local_file: Path, remote_path: str) -> None:
    """Upload a large file in 250 MB chunks and reassemble + SHA-256 verify on Colab."""
    size = local_file.stat().st_size
    if size <= CHUNK_BYTES:
        run([COLAB, "upload", "--session", session, str(local_file), remote_path], retries=3)
        return

    h = hashlib.sha256()
    chunk_dir = TMP_D / f"_chunks_{local_file.parent.name}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    remote_tmp = f"/content/_chunks_{local_file.parent.name}"
    exec_bash(session, f"mkdir -p {remote_tmp} $(dirname {remote_path})", tag=f"mk_{local_file.parent.name}", timeout=120)

    parts = []
    with local_file.open("rb") as f:
        idx = 0
        while True:
            buf = f.read(CHUNK_BYTES)
            if not buf:
                break
            h.update(buf)
            part_path = chunk_dir / f"part_{idx:02d}"
            part_path.write_bytes(buf)
            parts.append(part_path)
            idx += 1

    expected_sha = h.hexdigest()
    print(f"Uploading {local_file.name} ({size/(1024*1024):.1f} MB) in {len(parts)} chunks (sha={expected_sha[:12]}) ...", flush=True)
    for part_path in parts:
        run([COLAB, "upload", "--session", session, str(part_path), f"{remote_tmp}/{part_path.name}"], retries=3)
        part_path.unlink(missing_ok=True)

    exec_bash(
        session,
        f"set -e; cat {remote_tmp}/part_* > {remote_path}; rm -rf {remote_tmp}; "
        f"ACTUAL=$(sha256sum {remote_path} | awk '{{print $1}}'); "
        f"echo \"Reassembled {remote_path}: $ACTUAL\"; "
        f"[ \"$ACTUAL\" = \"{expected_sha}\" ] || (echo 'SHA MISMATCH'; exit 1)",
        tag=f"join_{local_file.parent.name}",
        timeout=300,
    )
    shutil.rmtree(chunk_dir, ignore_errors=True)


def main() -> None:
    TMP_D.mkdir(parents=True, exist_ok=True)
    code_tar = TMP_D / "m5_all_code.tar"
    if code_tar.exists():
        code_tar.unlink()

    print("=== Packaging unified code + manifests + precomputed test captions (lightweight <30 MB) ===", flush=True)
    with tarfile.open(code_tar, "w") as tf:
        for fn in ("m5_eval_detectors_and_stress.py", "m5_eval_captioners_and_verifier.py"):
            tf.add(M5 / "scripts" / fn, arcname=f"scripts/{fn}")
        for fn in ("colab_setup_detector.py", "colab_setup_captioner.py", "detector_models.py", "train_detector.py", "pipeline_recovery.py", "train_captioner.py"):
            tf.add(M4 / "scripts" / fn, arcname=f"scripts/{fn}")
        for p in sorted((REPO / "src").rglob("*.py")):
            if "__pycache__" not in p.parts:
                tf.add(p, arcname=f"src/{p.relative_to(REPO / 'src').as_posix()}")
        for p in sorted((M4 / "evidence/portable_data").glob("*")):
            if p.is_file():
                tf.add(p, arcname=f"data/portable_data/{p.name}")
        tf.add(M4 / "evidence/captioner_runs/C06_rscama_frozen_clip/eval_test_captions.jsonl", arcname="inputs/c06_test_captions.jsonl")
        tf.add(M4 / "evidence/captioner_runs/C00_rscama_released/eval_test_captions.jsonl", arcname="inputs/c00_test_captions.jsonl")
        tf.add(M4 / "evidence/verification/verification_claims.jsonl", arcname="inputs/val_claims.jsonl")
    print(f"Code bundle ready: {code_tar} ({code_tar.stat().st_size/(1024*1024):.1f} MB)", flush=True)

    started = False
    try:
        # Check if SESSION already exists & is ready
        chk = subprocess.run([COLAB, "sessions"], capture_output=True, text=True)
        if f"[{SESSION}]" in chk.stdout:
            print(f"=== REUSING EXISTING LIVE COLAB SESSION '{SESSION}' ===", flush=True)
            started = True
        else:
            for round_i in range(1, 5):
                for gpu in ("T4", "L4", "A100"):
                    r = run([COLAB, "new", "--session", SESSION, "--gpu", gpu], check=False)
                    if r.returncode == 0:
                        started = True
                        print(f"=== ACQUIRED COLAB {gpu} SESSION '{SESSION}' ===", flush=True)
                        break
                    time.sleep(3)
                if started:
                    break
                print(f"All GPUs transiently busy (round {round_i}/4), waiting 20s ...", flush=True)
                time.sleep(20)
        assert started, "Could not acquire T4, L4, or A100 GPU session"

        run([COLAB, "upload", "--session", SESSION, str(code_tar), "/content/m5_all_code.tar"], retries=3)
        code_tar.unlink(missing_ok=True)
        exec_bash(
            SESSION,
            "set -e; mkdir -p /content/terrawatch/checkpoints/D03_bit_lr3e4 /content/terrawatch/checkpoints/A03_siam_r18_unet "
            "/content/terrawatch/checkpoints/A01_ef_unet /content/terrawatch/checkpoints/A02_fc_siam_diff "
            "/content/terrawatch/checkpoints/A04_changeformer /content/terrawatch/checkpoints/C06 "
            "/content/terrawatch/checkpoints/C01 /content/terrawatch/checkpoints/C04; "
            "tar -xf /content/m5_all_code.tar -C /content/terrawatch; rm -f /content/m5_all_code.tar",
            tag="unpack",
            timeout=300,
        )

        # Non-interactive 45 MB chunked uploads (zero OAuth prompts, 100% below Jupyter single-POST limit!)
        for run_id in ("D03_bit_lr3e4", "A03_siam_r18_unet", "A01_ef_unet", "A02_fc_siam_diff", "A04_changeformer"):
            ckpt = M4 / "runs/detector" / run_id / "best.pt"
            upload_chunked(SESSION, ckpt, f"/content/terrawatch/checkpoints/{run_id}/best.pt")
        # Upload C06/best.pth in 45 MB chunks so the 200-pair captioner robustness probe runs cleanly!
        upload_chunked(SESSION, M4 / "runs/captioner/C06_rscama_frozen_clip/best.pth", "/content/terrawatch/checkpoints/C06/best.pth")

        # Bootstrap LEVIR-MCI, BIT_CD, ChangeFormer, RSCaMa, CLIP, bert-score, qwen-vl-utils
        exec_bash(
            SESSION,
            "set -e; BASE=/content/terrawatch; "
            "python3 -m pip install -q einops timm ftfy regex nltk scipy bert-score qwen-vl-utils accelerate; "
            "python3 $BASE/scripts/colab_setup_detector.py; "
            "python3 $BASE/scripts/colab_setup_captioner.py",
            tag="bootstrap_all",
            timeout=1800,
        )

        # PART A: Detector Test Full Metrics + 17-Condition Stress Suite + WS4-Det Error Mining
        exec_bash(
            SESSION,
            "set -e; BASE=/content/terrawatch; "
            "python3 $BASE/scripts/m5_eval_detectors_and_stress.py "
            "--manifests-dir $BASE/data/portable_data "
            "--extracted-root $BASE/data/extracted "
            "--checkpoints-dir $BASE/checkpoints "
            "--bit-source-dir $BASE/models/external/BIT_CD "
            "--changeformer-source-dir $BASE/models/external/ChangeFormer "
            "--output-dir $BASE/m5_det_out; "
            "tar -czf /content/m5_det_out.tar.gz -C $BASE/m5_det_out . && ls -lh /content/m5_det_out.tar.gz",
            tag="part_a_det",
            timeout=5400,
        )

        det_tar = TMP_D / "m5_det_out.tar.gz"
        run([COLAB, "download", "--session", SESSION, "/content/m5_det_out.tar.gz", str(det_tar)], retries=3)
        dst_det = M5 / "evidence/test_detector_and_stress"
        if dst_det.exists():
            shutil.rmtree(dst_det)
        dst_det.mkdir(parents=True, exist_ok=True)
        with tarfile.open(det_tar, "r:gz") as tf:
            tf.extractall(dst_det)
        det_tar.unlink(missing_ok=True)
        print(f"=== PART A PULLED -> {dst_det} ===", flush=True)

        # PART B: Captioners (1,858 Test + BERTScore/CIDEr/METEOR + 200 Probe) + Blind 150-Pair Test Hybrid Verifier + 120-Claim Blind Packet
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
            tag="part_b_cap_ver",
            timeout=7200,
        )

        cap_tar = TMP_D / "m5_cap_ver_out.tar.gz"
        run([COLAB, "download", "--session", SESSION, "/content/m5_cap_ver_out.tar.gz", str(cap_tar)], retries=3)
        dst_cap = M5 / "evidence/test_captioner_and_verifier"
        if dst_cap.exists():
            shutil.rmtree(dst_cap)
        dst_cap.mkdir(parents=True, exist_ok=True)
        with tarfile.open(cap_tar, "r:gz") as tf:
            tf.extractall(dst_cap)
        cap_tar.unlink(missing_ok=True)
        print(f"=== PART B PULLED -> {dst_cap} ===", flush=True)
        print("=== ALL MILESTONE 5 COLAB SUITES COMPLETE ===", flush=True)
    finally:
        if started:
            run([COLAB, "stop", "--session", SESSION], check=False)
        shutil.rmtree(TMP_D, ignore_errors=True)


if __name__ == "__main__":
    main()
