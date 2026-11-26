"""WSL orchestrator for Colab L4 #1 (m5-det): WS1-Det (5 detectors + controls on 1,858 Test pairs) + WS5 (17-condition Test stress suite) + WS4-Det error mining."""
from __future__ import annotations

import shutil
import subprocess
import tarfile
import time
from pathlib import Path

COLAB = "/home/shoaib/.local/bin/colab"
REPO = Path("/mnt/d/IITM/DegreeTerm3/DSAI/Project")
M4 = REPO / "POC/milestone4/rework"
M5 = REPO / "POC/milestone5"
TMP_D = M5 / "tmp_det"
SESSION = "m5-det"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def exec_bash(session: str, bash_cmd: str, tag: str, timeout: int = 5400) -> None:
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
    bundle = TMP_D / "m5_det_inputs.tar"
    if bundle.exists():
        bundle.unlink()

    print("=== Packaging m5-det bundle (scripts + portable_data + 5 detector best.pt) ===", flush=True)
    with tarfile.open(bundle, "w") as tf:
        tf.add(M5 / "scripts/m5_eval_detectors_and_stress.py", arcname="scripts/m5_eval_detectors_and_stress.py")
        for fn in ("colab_setup_detector.py", "detector_models.py", "train_detector.py"):
            tf.add(M4 / "scripts" / fn, arcname=f"scripts/{fn}")
        for p in sorted((M4 / "evidence/portable_data").glob("*")):
            if p.is_file():
                tf.add(p, arcname=f"data/portable_data/{p.name}")
        for run_id in ("D03_bit_lr3e4", "A03_siam_r18_unet", "A01_ef_unet", "A02_fc_siam_diff", "A04_changeformer"):
            ckpt = M4 / "runs/detector" / run_id / "best.pt"
            assert ckpt.is_file(), f"Missing {ckpt}"
            tf.add(ckpt, arcname=f"checkpoints/{run_id}/best.pt")
    print(f"Bundle ready: {bundle} ({bundle.stat().st_size/(1024*1024):.1f} MB)", flush=True)

    started = False
    try:
        for gpu in ("L4", "T4"):
            r = run([COLAB, "new", "--session", SESSION, "--gpu", gpu], check=False)
            if r.returncode == 0:
                started = True
                break
        assert started, "Could not acquire L4 or T4 GPU session"
        run([COLAB, "upload", "--session", SESSION, str(bundle), "/content/m5_det_inputs.tar"])
        bundle.unlink(missing_ok=True)

        exec_bash(
            SESSION,
            "set -e; mkdir -p /content/terrawatch; tar -xf /content/m5_det_inputs.tar -C /content/terrawatch; "
            "rm -f /content/m5_det_inputs.tar; "
            "python3 -m pip install -q scipy; "
            "python3 /content/terrawatch/scripts/colab_setup_detector.py",
            tag="setup_det",
            timeout=1800,
        )

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
            tag="run_det",
            timeout=5400,
        )

        out_tar = TMP_D / "m5_det_out.tar.gz"
        if out_tar.exists():
            out_tar.unlink()
        run([COLAB, "download", "--session", SESSION, "/content/m5_det_out.tar.gz", str(out_tar)])

        dst = M5 / "evidence/test_detector_and_stress"
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        with tarfile.open(out_tar, "r:gz") as tf:
            tf.extractall(dst)
        out_tar.unlink(missing_ok=True)
        print(f"=== M5-DET COMPLETE -> {dst} ===", flush=True)
    finally:
        if started:
            run([COLAB, "stop", "--session", SESSION], check=False)
        shutil.rmtree(TMP_D, ignore_errors=True)


if __name__ == "__main__":
    main()
