"""Upgraded local Evidence Review server for TerraWatch Milestone 4 recovery.

Loads immutable pipeline run folders (report.json + review_inputs.json + assets),
verifies SHA-256 hashes of source images on load, preserves original model decisions
immutably, and records human flags/corrections/decisions in an append-only SQLite audit log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlsplit

APP_DIR = Path(__file__).resolve().parent


def load_reports(folders: List[Path]) -> Tuple[Dict[str, dict], Dict[str, Path]]:
    records: Dict[str, dict] = {}
    assets: Dict[str, Path] = {}

    def register(path_val: str | Path, folder: Path, run_id: str) -> dict:
        p = Path(path_val)
        if not p.is_absolute():
            p = folder / p
        p = p.resolve()
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"Missing or unsupported image asset: {p}")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        key = f"{run_id[:16]}-{digest[:24]}"
        assets[key] = p
        return {"url": f"/asset/{key}", "sha256": digest, "filename": p.name}

    for raw_folder in folders:
        folder = Path(raw_folder).resolve()
        rep_path = folder / "report.json"
        if not rep_path.is_file():
            raise ValueError(f"Missing report.json in {folder}")
        raw_bytes = rep_path.read_bytes()
        run = json.loads(raw_bytes)
        run_id = hashlib.sha256(raw_bytes).hexdigest()

        sidecar = folder / "review_inputs.json"
        if sidecar.exists():
            inp = json.loads(sidecar.read_text(encoding="utf-8"))
            images = inp["images"]
            mask = inp["mask"]
            overlay = inp.get("overlay")
        else:
            images = json.loads((folder / "detector_request.json").read_text(encoding="utf-8"))["images"]
            resp = json.loads((folder / "detector_response.json").read_text(encoding="utf-8"))
            mask = folder / "detector_response.mask.png"
            if not mask.exists():
                mask = Path(resp["mask_path"])
            overlay = None

        originals = [register(p, folder, run_id) for p in images]
        if [img["sha256"] for img in originals] != run["input_sha256"]:
            raise ValueError(f"Input image SHA-256 mismatch against {rep_path}")
        mask_asset = register(mask, folder, run_id)
        overlay_asset = register(overlay, folder, run_id) if overlay and (folder / overlay).is_file() else None

        for claim in run.get("claims", []):
            record_id = f"{run_id[:16]}:{claim['claim_id']}"
            crops = []
            for cp in claim.get("evidence", {}).get("crop_paths", []):
                portable = folder / "crops" / Path(str(cp).replace("\\", "/")).name
                crops.append(register(portable if portable.exists() else cp, folder, run_id))
            records[record_id] = {
                "id": record_id,
                "run_id": run_id,
                "folder": folder.name,
                "trace_category": run.get("trace_category", ""),
                "stratum": run.get("stratum", ""),
                "execution_mode": run.get("execution_mode", "model_workers"),
                "pair_id": run["pair_id"],
                "generated_caption": run.get("generated_caption", ""),
                "verified_description": run.get("verified_description", ""),
                "claim": claim,
                "images": originals,
                "mask": mask_asset,
                "overlay": overlay_asset,
                "crops": crops,
                "model_metadata": run.get("model_metadata", {}),
            }

    if not records:
        raise ValueError("No claim records found in the selected folders")
    return records, assets


class ReviewStore:
    """Append-only SQLite store preserving every reviewer event alongside immutable model provenance."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS reviews ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "claim_id TEXT NOT NULL, "
                "event TEXT NOT NULL)"
            )

    def list(self, record_id: str) -> List[dict]:
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("SELECT event FROM reviews WHERE claim_id=? ORDER BY id", (record_id,)).fetchall()
            return [json.loads(r[0]) for r in rows]

    def append(self, record: dict, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object")
        reviewer = payload.get("reviewer")
        notes = payload.get("notes", "")
        correction = payload.get("correction", "")
        decision = payload.get("decision")
        flagged = payload.get("flagged", False)

        if not isinstance(reviewer, str) or not reviewer.strip() or len(reviewer) > 120:
            raise ValueError("Reviewer name is required (1-120 chars)")
        if decision not in (None, "accept", "reject", "defer") or type(flagged) is not bool:
            raise ValueError("Invalid decision or flagged value")
        if any(not isinstance(s, str) or len(s) > 2000 for s in (notes, correction)):
            raise ValueError("Notes and correction must be strings <= 2000 chars")

        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT event FROM reviews WHERE claim_id=? ORDER BY id DESC LIMIT 1", (record["id"],)
            ).fetchone()
            prev = json.loads(row[0]) if row else None
            orig_outcome = record["claim"]["decision"]["outcome"]
            event = {
                "record_id": record["id"],
                "claim_id": record["claim"]["claim_id"],
                "pair_id": record["pair_id"],
                "run_id": record["run_id"],
                "reviewer": reviewer.strip(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "decision": decision,
                "is_human_override": bool(decision is not None and decision != orig_outcome),
                "flagged": flagged,
                "correction": correction.strip(),
                "notes": notes.strip(),
                "original_model_outcome": orig_outcome,
                "previous_review": (
                    {k: prev.get(k) for k in ("decision", "flagged", "correction", "reviewer", "timestamp")}
                    if prev
                    else None
                ),
            }
            conn.execute("INSERT INTO reviews(claim_id, event) VALUES (?, ?)", (record["id"], json.dumps(event)))
        return event


def make_server(records: Dict[str, dict], assets: Dict[str, Path], store: ReviewStore, port: int = 8765):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code: int, body: bytes, mime: str):
            self.send_response(code)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj):
            self._send(code, json.dumps(obj, indent=2).encode("utf-8"), "application/json")

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self._send(200, (APP_DIR / "review.html").read_bytes(), "text/html; charset=utf-8")
            elif path in ("/api/reports", "/api/export"):
                payload = [{**rec, "reviews": store.list(k)} for k, rec in records.items()]
                self._json(200, payload)
            elif path.startswith("/asset/") and path[7:] in assets:
                p = assets[path[7:]]
                mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
                self._send(200, p.read_bytes(), mime)
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/review":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                if self.headers.get_content_type() != "application/json":
                    raise ValueError("Content-Type must be application/json")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or data.get("id") not in records:
                    raise ValueError("Unknown claim record id")
                event = store.append(records[data["id"]], data)
            except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})
                return
            self._json(201, event)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def discover_run_folders(paths: List[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if (p / "report.json").is_file():
            found.append(p)
        elif p.is_dir():
            for child in sorted(p.iterdir()):
                if child.is_dir() and (child / "report.json").is_file():
                    found.append(child)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="Run directories (or a parent folder containing run directories)")
    ap.add_argument("--db", default="outputs/logs/review_audit.sqlite3")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    folders = discover_run_folders(args.runs)
    records, assets = load_reports(folders)
    store = ReviewStore(args.db)
    srv = make_server(records, assets, store, args.port)
    print(f"TerraWatch Evidence Review listening at http://127.0.0.1:{args.port} ({len(records)} claims from {len(folders)} runs)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
