"""Lightweight, fast Colab T4/L4 runner for:
  1. Part B (m5_eval_captioners_and_verifier.py): ALL 6 captioners (C06, C00, C04, C01, BIT-Template, Always-NC)
     on 1,858 held-out Test pairs with BLEU-1..4, ROUGE-L, METEOR, CIDEr-D, BERTScore-F1 (roberta-large),
     200-pair corruption probe scoring, AND 150-pair blind Test Hybrid Verifier (Qwen3-VL-4B-Instruct)
     + 10 Test E2E traces + 120-claim blind human-audit packet (M5T_001..M5T_120) + visual strips!
  2. Sujal's m5_benchmark_compute.py on Colab GPU:
     - colab_t4_referee (Qwen3-VL-4B-Instruct, warmup=3, iters=20, transformers 5.x)
     - colab_t4 (all 5 real detectors + OpenCV witness)
     - colab_t4_cap (C06 captioner ref + Triton scan via isolated --target /tmp/tf439 transformers==4.39.3!)
     (automatically labels colab_l4* if an L4 GPU is allocated!)

ZERO 1.67 GB uploads (except optional C06 if benchmarking captioner on Colab; detectors are only ~330 MB in 45 MB chunks!).
Always stops the Colab session cleanly in finally:.
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
TMP_D = M5 / "tmp_partb"
SESSION = "m5-partb"
CHUNK_BYTES = 45 * 1024 * 1024


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


def exec_bash(session: str, bash_cmd: str, tag: str, timeout: int = 7200) -> str:
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
    return proc.stdout


def upload_chunked(session: str, local_file: Path, remote_path: str) -> None:
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
    for part_path in parts:
        run([COLAB, "upload", "--session", session, str(part_path), f"{remote_tmp}/{part_path.name}"], retries=3)
        part_path.unlink(missing_ok=True)
    exec_bash(
        session,
        f"set -e; cat {remote_tmp}/part_* > {remote_path}; rm -rf {remote_tmp}; "
        f"ACTUAL=$(sha256sum {remote_path} | awk '{{print $1}}'); "
        f"[ \"$ACTUAL\" = \"{expected_sha}\" ] || (echo 'SHA MISMATCH'; exit 1)",
        tag=f"join_{local_file.parent.name}",
        timeout=300,
    )
    shutil.rmtree(chunk_dir, ignore_errors=True)


def main() -> None:
    TMP_D.mkdir(parents=True, exist_ok=True)
    code_tar = TMP_D / "m5_partb_code.tar"
    if code_tar.exists():
        code_tar.unlink()

    with tarfile.open(code_tar, "w") as tf:
        tf.add(M5 / "scripts/m5_eval_captioners_and_verifier.py", arcname="scripts/m5_eval_captioners_and_verifier.py")
        for fn in ("colab_setup_detector.py", "detector_models.py", "pipeline_recovery.py"):
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
        for opt_name in ("C01_rscama_ref_test_captions.jsonl", "C04_rscama_adamw_cos_test_captions.jsonl", "c06_probe200_captions.json"):
            opt_p = M5 / "evidence/omen_captions" / opt_name
            if opt_p.is_file():
                tf.add(opt_p, arcname=f"inputs/{opt_name}")
    print(f"Part B code bundle ready: {code_tar} ({code_tar.stat().st_size/(1024*1024):.1f} MB)", flush=True)

    started = False
    gpu_label = "colab_t4"
    try:
        for round_i in range(1, 5):
            for gpu in ("T4", "L4"):
                r = run([COLAB, "new", "--session", SESSION, "--gpu", gpu], check=False)
                if r.returncode == 0:
                    started = True
                    gpu_label = "colab_l4" if gpu == "L4" else "colab_t4"
                    print(f"=== ACQUIRED {gpu} -> label={gpu_label} ===", flush=True)
                    break
                time.sleep(3)
            if started:
                break
            time.sleep(15)
        assert started, "Could not acquire T4 or L4"

        run([COLAB, "upload", "--session", SESSION, str(code_tar), "/content/m5_partb_code.tar"], retries=3)
        code_tar.unlink(missing_ok=True)
        exec_bash(
            SESSION,
            "set -e; mkdir -p /content/terrawatch/checkpoints/D03_bit_lr3e4 /content/terrawatch/checkpoints/A03_siam_r18_unet "
            "/content/terrawatch/checkpoints/A01_ef_unet /content/terrawatch/checkpoints/A02_fc_siam_diff "
            "/content/terrawatch/checkpoints/A04_changeformer; "
            "tar -xf /content/m5_partb_code.tar -C /content/terrawatch; rm -f /content/m5_partb_code.tar",
            tag="unpack",
            timeout=300,
        )

        # Upload all 5 detector checkpoints (total 330 MB, ~45 sec!)
        for run_id in ("D03_bit_lr3e4", "A03_siam_r18_unet", "A01_ef_unet", "A02_fc_siam_diff", "A04_changeformer"):
            ckpt = M4 / "runs/detector" / run_id / "best.pt"
            upload_chunked(SESSION, ckpt, f"/content/terrawatch/checkpoints/{run_id}/best.pt")

        # Bootstrap LEVIR-MCI, BIT_CD, ChangeFormer, bert-score, qwen-vl-utils, and Sujal's repo branch
        exec_bash(
            SESSION,
            "set -e; BASE=/content/terrawatch; "
            "python3 -m pip install -q einops timm nltk scipy bert-score qwen-vl-utils accelerate ftfy regex; "
            "python3 $BASE/scripts/colab_setup_detector.py; "
            "if [ ! -d /content/repo_m5 ]; then git clone --depth 1 -b m5-compute-and-audit https://github.com/shoaib2000857/Group-7-DS-and-AI-Lab-Project.git /content/repo_m5; fi; "
            "mkdir -p /content/repo_m5/models /content/repo_m5/POC/execution/data /content/bench_out; "
            "ln -sfn $BASE/models/external /content/repo_m5/models/external; "
            "rm -rf /content/repo_m5/POC/execution/data/pairs; "
            "ln -sfn $BASE/data/extracted/LEVIR-MCI-dataset/images /content/repo_m5/POC/execution/data/pairs",
            tag="bootstrap",
            timeout=1800,
        )

        # 1. Run Part B: All 6 captioners + BERTScore/CIDEr/METEOR + 200-pair probe + 150-pair blind Test Hybrid Verifier + 120-claim blind audit packet
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
            "--c01-test-captions $BASE/inputs/C01_rscama_ref_test_captions.jsonl "
            "--c04-test-captions $BASE/inputs/C04_rscama_adamw_cos_test_captions.jsonl "
            "--probe-captions-json $BASE/inputs/c06_probe200_captions.json "
            "--val-claims-jsonl $BASE/inputs/val_claims.jsonl "
            "--output-dir $BASE/m5_cap_ver_out; "
            "tar -czf /content/m5_cap_ver_out.tar.gz -C $BASE/m5_cap_ver_out . && ls -lh /content/m5_cap_ver_out.tar.gz",
            tag="part_b",
            timeout=5400,
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

        # 2. Run Sujal's m5_benchmark_compute.py for detectors, witness, AND Qwen3-VL referee on this exact Colab GPU!
        exec_bash(
            SESSION,
            f"set -e; cd /content/repo_m5; "
            f"python3 POC/milestone5/scripts/m5_benchmark_compute.py --device cuda --hardware-label {gpu_label} "
            f"--ckpt-dir /content/terrawatch/checkpoints --stages detectors,witness --out /content/bench_out/; "
            f"python3 POC/milestone5/scripts/m5_benchmark_compute.py --device cuda --hardware-label {gpu_label}_referee "
            f"--stages referee --stage-overrides 'referee:warmup=3,iters=20' --out /content/bench_out/",
            tag="sujal_bench",
            timeout=1800,
        )

        # 3. Upload C06 checkpoint (45 MB chunks) and run Sujal's captioner ref + Triton benchmark on Colab GPU!
        c06_ckpt = M4 / "runs/captioner/C06_rscama_frozen_clip/best.pth"
        exec_bash(SESSION, "mkdir -p /content/terrawatch/checkpoints/C06_rscama_frozen_clip", tag="mk_c06", timeout=120)
        upload_chunked(SESSION, c06_ckpt, "/content/terrawatch/checkpoints/C06_rscama_frozen_clip/best.pth")
        exec_bash(
            SESSION,
            f"set -e; mkdir -p /tmp/tf439; "
            f"python3 -m pip install -q --no-deps --target /tmp/tf439 transformers==4.39.3; "
            f"printf 'def dep_version_check(*a, **k):\\n    return None\\n' > /tmp/tf439/transformers/dependency_versions_check.py; "
            f"if [ ! -d /content/terrawatch/models/external/RSCaMa ]; then "
            f"  git clone -q https://github.com/Chen-Yang-Liu/RSCaMa.git /content/terrawatch/models/external/RSCaMa && "
            f"  git -C /content/terrawatch/models/external/RSCaMa checkout -q aa68e4c6933c; "
            f"fi; "
            f"if [ ! -d /content/terrawatch/models/external/CLIP ]; then "
            f"  git clone -q https://github.com/openai/CLIP.git /content/terrawatch/models/external/CLIP && "
            f"  git -C /content/terrawatch/models/external/CLIP checkout -q dcba3cb2e282; "
            f"fi; "
            f"cd /content/repo_m5; "
            f"python3 -c 'p=\"POC/milestone5/scripts/m5_benchmark_compute.py\"; s=open(p).read().replace(\", vocab, \\\"cpu\\\")\", \", vocab, device)\"); open(p,\"w\").write(s)'; "
            f"PYTHONPATH=/tmp/tf439:/content/terrawatch/models/external/CLIP python3 POC/milestone5/scripts/m5_benchmark_compute.py "
            f"--device cuda --hardware-label {gpu_label}_cap --ckpt-dir /content/terrawatch/checkpoints "
            f"--stages captioner --stage-overrides 'captioner:warmup=5,iters=30' --out /content/bench_out/; "
            f"tar -czf /content/colab_bench.tar.gz -C /content/bench_out . && ls -lh /content/colab_bench.tar.gz",
            tag="sujal_bench_cap",
            timeout=1800,
        )

        b_tar = TMP_D / "colab_bench.tar.gz"
        run([COLAB, "download", "--session", SESSION, "/content/colab_bench.tar.gz", str(b_tar)], retries=3)
        dst_b = M5 / "evidence/compute_profile"
        dst_b.mkdir(parents=True, exist_ok=True)
        with tarfile.open(b_tar, "r:gz") as tf:
            tf.extractall(dst_b)
        b_tar.unlink(missing_ok=True)
        print(f"=== SUJAL COLAB BENCHMARK PULLED -> {dst_b} ===", flush=True)
    finally:
        if started:
            run([COLAB, "stop", "--session", SESSION], check=False)
        shutil.rmtree(TMP_D, ignore_errors=True)


if __name__ == "__main__":
    main()
