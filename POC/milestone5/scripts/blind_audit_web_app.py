"""TerraWatch Milestone 5 — Blind Dual-Rater Human Audit Web App (Pure Python stdlib).

Serves a fast, keyboard-driven blind annotation interface over the 120 held-out Test claims
(M5T_001..M5T_120) with ZERO model verdicts, probabilities, or cross-rater labels exposed.

Saves every annotation atomically to `rater_<INITIALS>.csv` in Sujal's exact schema for
`POC/milestone5/scripts/m5_compute_human_audit.py`.

Usage:
    python blind_audit_web_app.py --data-dir /path/to/blind_audit_web --port 8770
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RATERS = {
    "SSSM": {"name": "Shoaib Sayyed (SSSM)", "file": "rater_SSSM_60.csv", "slices": "001–030 & 091–120"},
    "PA": {"name": "Priyanshu Agarwal (PA)", "file": "rater_PA_60.csv", "slices": "001–060"},
    "SS": {"name": "Sarthak Sharma (SS)", "file": "rater_SS_60.csv", "slices": "031–090"},
    "SRP": {"name": "Sujal Pradhan (SRP)", "file": "rater_SRP_60.csv", "slices": "061–120"},
}

ALLOWED_LABELS = ("supported", "contradicted", "unverifiable")
ALLOWED_CATS = (
    "none",
    "hallucinated_object",
    "missed_change",
    "wrong_class_road_vs_bldg",
    "wrong_count",
    "wrong_location",
    "shadow_or_seasonal",
    "tiny_or_occluded",
)
CSV_FIELDS = [
    "audit_id", "pair_id", "stratum", "claim_id", "caption", "claim_span",
    "object", "action", "location", "quantity", "human_label", "error_category", "notes",
]

LOCK = threading.Lock()
DATA_DIR: Path = Path(".")


def _rater_csv_path(rater: str) -> Path:
    out_dir = DATA_DIR / "saved_raters"
    out_dir.mkdir(parents=True, exist_ok=True)
    live = out_dir / f"rater_{rater}.csv"
    if not live.is_file():
        src = DATA_DIR / RATERS[rater]["file"]
        live.write_bytes(src.read_bytes())
    return live


def load_rater_rows(rater: str) -> list[dict]:
    p = _rater_csv_path(rater)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def save_rater_rows(rater: str, rows: list[dict]) -> None:
    p = _rater_csv_path(rater)
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    os.replace(tmp, p)


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>TerraWatch M5 — Blind Human Claim Audit</title>
<style>
  :root {
    --navy: #0e222c; --teal: #1b6b74; --lime: #8bbf2a; --bg: #f4f6f5;
    --card: #ffffff; --ink: #1a2830; --muted: #5d7079; --border: #d5dfe3;
    --sup: #1f8a5b; --con: #c93b3b; --unv: #c97f1d;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--ink); }
  header {
    background: var(--navy); color: #f5f7f6; padding: 12px 22px; display: flex;
    align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap;
    border-bottom: 3px solid var(--teal);
  }
  header h1 { margin: 0; font-size: 17px; letter-spacing: 0.3px; }
  header .sub { font-size: 12px; color: #b7ccd3; }
  .pill { background: rgba(255,255,255,0.12); padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .container { max-width: 1240px; margin: 18px auto; padding: 0 16px 40px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; box-shadow: 0 2px 8px rgba(16,38,48,0.05); }
  .grid-4 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 14px; }
  .rater-card {
    border: 1.5px solid var(--border); border-radius: 10px; padding: 16px; cursor: pointer;
    transition: all 0.15s; background: #fff; text-align: left;
  }
  .rater-card:hover { border-color: var(--teal); transform: translateY(-2px); box-shadow: 0 6px 16px rgba(27,107,116,0.12); }
  .rater-card h3 { margin: 0 0 4px; font-size: 16px; color: var(--navy); }
  .rater-card .meta { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; }
  .bar-wrap { height: 8px; background: #e5ecef; border-radius: 99px; overflow: hidden; margin-bottom: 6px; }
  .bar-fill { height: 100%; background: var(--teal); transition: width 0.25s; }
  .btn {
    border: none; border-radius: 7px; padding: 9px 14px; font-size: 13.5px; font-weight: 600;
    cursor: pointer; transition: opacity 0.15s, transform 0.08s; display: inline-flex; align-items: center; gap: 6px;
  }
  .btn:hover { opacity: 0.92; }
  .btn:active { transform: scale(0.98); }
  .btn-navy { background: var(--navy); color: #fff; }
  .btn-ghost { background: #e8eff1; color: var(--navy); }
  .btn-sup { background: var(--sup); color: #fff; font-size: 15px; padding: 12px 18px; }
  .btn-con { background: var(--con); color: #fff; font-size: 15px; padding: 12px 18px; }
  .btn-unv { background: var(--unv); color: #fff; font-size: 15px; padding: 12px 18px; }
  .btn.selected { outline: 3px solid var(--navy); outline-offset: 2px; }
  .strip-box { background: #102630; border-radius: 8px; padding: 8px; text-align: center; margin-bottom: 14px; }
  .strip-box img { max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto; }
  .claim-banner {
    background: #eef6f7; border-left: 4px solid var(--teal); padding: 12px 15px; border-radius: 6px;
    margin-bottom: 14px;
  }
  .claim-span { font-size: 18px; font-weight: 700; color: var(--navy); margin-bottom: 5px; }
  .claim-meta { font-size: 12.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 10px; }
  .tag { background: #dce8eb; color: #16444c; padding: 2px 8px; border-radius: 4px; font-family: Consolas, monospace; font-size: 12px; }
  .controls-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }
  .cats-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 14px; }
  .cat-btn {
    border: 1.5px solid var(--border); background: #fff; color: var(--ink); padding: 7px 11px;
    border-radius: 6px; font-size: 12.5px; font-weight: 600; cursor: pointer;
  }
  .cat-btn:hover { border-color: var(--teal); }
  .cat-btn.active { background: var(--navy); color: #fff; border-color: var(--navy); }
  .nav-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(46px, 1fr)); gap: 5px; margin-top: 14px; }
  .nav-dot {
    border: 1px solid var(--border); background: #fff; border-radius: 5px; padding: 5px 0;
    font-size: 11px; font-weight: 700; cursor: pointer; text-align: center;
  }
  .nav-dot.cur { outline: 2.5px solid var(--navy); }
  .nav-dot.st-supported { background: #d6f1e4; color: #145e3d; border-color: #96d8b9; }
  .nav-dot.st-contradicted { background: #fadede; color: #8c2020; border-color: #ebb0b0; }
  .nav-dot.st-unverifiable { background: #fcecd4; color: #87500b; border-color: #ecc691; }
  .rubric { font-size: 12.5px; color: var(--muted); line-height: 1.5; background: #fafbfa; border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px; margin-top: 14px; }
  input[type="text"] { width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 6px; font-size: 13px; }
  .toast { position: fixed; bottom: 18px; right: 18px; background: var(--navy); color: #d9ed85; padding: 9px 15px; border-radius: 8px; font-size: 13px; font-weight: 600; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.2); }
</style>
</head>
<body>
<header>
  <div>
    <h1>TerraWatch Milestone 5 — Blind Human Claim Audit</h1>
    <div class="sub">IIT Madras BS DSAI Lab (Group 7) • Zero model verdicts shown • Auto-saves directly to Sujal's CSV schema</div>
  </div>
  <div style="display:flex; gap:8px; align-items:center;">
    <span class="pill" id="hdrStatus">Select Rater</span>
    <button class="btn btn-ghost" id="btnHome" style="display:none; padding:6px 11px; font-size:12px;" onclick="showHome()">Switch Rater</button>
    <a class="btn btn-ghost" id="btnDownload" style="display:none; padding:6px 11px; font-size:12px; text-decoration:none;" href="#">Download CSV</a>
  </div>
</header>

<div class="container">
  <!-- HOME VIEW -->
  <div id="homeView" class="card">
    <h2 style="margin:0 0 6px; color:var(--navy);">Choose Your Rater Sheet (60 Claims Each)</h2>
    <p style="margin:0 0 10px; color:var(--muted); font-size:14px;">
      Inspect <strong>T1 (Before)</strong>, <strong>T2 (After)</strong>, and the <strong>Zoom Crops</strong> and judge whether the atomic claim is visually <strong>Supported</strong>, <strong>Contradicted</strong>, or <strong>Unverifiable</strong>. Do not discuss labels with your paired rater until finished.
    </p>
    <div class="grid-4" id="raterCards"></div>
    <div class="rubric">
      <strong>Quick 3-Class Rubric (Appendix J):</strong><br/>
      • <strong>Supported [Key 1]:</strong> The claimed change (or lack of change) is clearly visible in T1 vs T2, and object type/location match. (Sets category = <code>none</code>.)<br/>
      • <strong>Contradicted [Key 2]:</strong> The images visibly disagree with the claim (e.g. claims a new house/road that isn't there, says "no change" when a real change happened, wrong class road vs building, wrong count, or wrong location).<br/>
      • <strong>Unverifiable [Key 3]:</strong> Cannot be confirmed visually (too tiny/blurry/occluded, heavy shadow/seasonal ambiguity, or out-of-scope detail).<br/>
      <em>Tip:</em> Use keys <code>1</code> / <code>2</code> / <code>3</code> and <code>←</code> / <code>→</code> arrows to fly through all 60 claims in ~10 minutes!
    </div>
  </div>

  <!-- WORKSPACE VIEW -->
  <div id="workView" style="display:none;">
    <div class="card">
      <div class="controls-row" style="justify-content:space-between;">
        <div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">
          <button class="btn btn-ghost" onclick="step(-1)">← Prev</button>
          <button class="btn btn-ghost" onclick="step(1)">Next →</button>
          <button class="btn btn-navy" onclick="jumpNextUnlabelled()">Next Unlabelled ⏭</button>
        </div>
        <div style="font-size:13.5px; font-weight:700; color:var(--navy);" id="posLabel">Claim 1 / 60</div>
      </div>

      <div class="claim-banner">
        <div class="claim-span" id="claimSpan"></div>
        <div class="claim-meta" id="claimMeta"></div>
      </div>

      <div class="strip-box">
        <img id="stripImg" src="" alt="Blind T1/T2 inspection strip" />
      </div>

      <div style="margin-bottom:8px; font-size:13px; font-weight:700; color:var(--navy);">1. Select Visual Verdict (Keys 1 / 2 / 3):</div>
      <div class="controls-row">
        <button class="btn btn-sup" id="lbl_supported" onclick="pickLabel('supported')">[1] Supported ✓</button>
        <button class="btn btn-con" id="lbl_contradicted" onclick="pickLabel('contradicted')">[2] Contradicted ✗</button>
        <button class="btn btn-unv" id="lbl_unverifiable" onclick="pickLabel('unverifiable')">[3] Unverifiable ?</button>
        <label style="margin-left:auto; font-size:12.5px; color:var(--muted); display:flex; align-items:center; gap:5px; cursor:pointer;">
          <input type="checkbox" id="autoAdv" checked /> Auto-advance on save
        </label>
      </div>

      <div id="catWrap" style="display:none;">
        <div style="font-size:13px; font-weight:700; color:var(--navy);">2. Pick Primary Error Category (required for Contradicted / Unverifiable):</div>
        <div class="cats-grid" id="catsGrid"></div>
      </div>

      <div style="display:flex; gap:10px; align-items:center; margin-top:8px;">
        <input type="text" id="notesInput" placeholder="Optional short note (press Enter to save note)" onkeydown="if(event.key==='Enter'){saveCurrent(false);}" />
        <button class="btn btn-navy" onclick="saveCurrent(true)">Save & Next</button>
      </div>

      <div style="margin-top:16px; font-size:12px; font-weight:700; color:var(--muted);">CLAIM MAP (Green=Supported, Red=Contradicted, Amber=Unverifiable, White=Pending):</div>
      <div class="nav-grid" id="navGrid"></div>
    </div>
  </div>
</div>

<div class="toast" id="toast">Saved ✓</div>

<script>
const CAT_DEFS = [
  ["hallucinated_object", "Hallucinated object (not in image)"],
  ["missed_change", "Missed real change"],
  ["wrong_class_road_vs_bldg", "Wrong class (road vs building)"],
  ["wrong_count", "Wrong count"],
  ["wrong_location", "Wrong location / relation"],
  ["shadow_or_seasonal", "Shadow / seasonal illusion"],
  ["tiny_or_occluded", "Tiny / blurry / occluded"],
];
let curRater = null;
let rows = [];
let idx = 0;
let pendingLabel = "";
let pendingCat = "none";

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg; t.style.display = "block";
  clearTimeout(window._tt); window._tt = setTimeout(() => t.style.display = "none", 1300);
}

async function loadStatus() {
  const res = await fetch("/api/status");
  const data = await res.json();
  const wrap = document.getElementById("raterCards");
  wrap.innerHTML = "";
  for (const r of data.raters) {
    const pct = Math.round((r.done / Math.max(r.total, 1)) * 100);
    const el = document.createElement("button");
    el.className = "rater-card";
    el.onclick = () => openRater(r.id);
    el.innerHTML = `
      <h3>${r.name}</h3>
      <div class="meta">Assigned claims: <strong>${r.slices}</strong></div>
      <div class="bar-wrap"><div class="bar-fill" style="width:${pct}%"></div></div>
      <div style="font-size:13px; font-weight:700; color:${pct===100?'var(--sup)':'var(--navy)'};">${r.done} / ${r.total} rated (${pct}%)</div>
    `;
    wrap.appendChild(el);
  }
}

function showHome() {
  curRater = null;
  document.getElementById("homeView").style.display = "block";
  document.getElementById("workView").style.display = "none";
  document.getElementById("btnHome").style.display = "none";
  document.getElementById("btnDownload").style.display = "none";
  document.getElementById("hdrStatus").textContent = "Select Rater";
  loadStatus();
}

async function openRater(raterId) {
  curRater = raterId;
  const res = await fetch(`/api/rows?rater=${encodeURIComponent(raterId)}`);
  rows = await res.json();
  document.getElementById("homeView").style.display = "none";
  document.getElementById("workView").style.display = "block";
  document.getElementById("btnHome").style.display = "inline-flex";
  const dl = document.getElementById("btnDownload");
  dl.style.display = "inline-flex";
  dl.href = `/api/download?rater=${encodeURIComponent(raterId)}`;
  // Jump to first unlabelled row automatically
  const firstBlank = rows.findIndex(r => !r.human_label);
  idx = firstBlank >= 0 ? firstBlank : 0;
  renderRow();
}

function renderNav() {
  const g = document.getElementById("navGrid");
  g.innerHTML = "";
  const done = rows.filter(r => r.human_label).length;
  document.getElementById("hdrStatus").textContent = `${curRater}: ${done}/${rows.length} done`;
  rows.forEach((r, i) => {
    const b = document.createElement("button");
    b.className = "nav-dot" + (i === idx ? " cur" : "") + (r.human_label ? " st-" + r.human_label : "");
    b.textContent = r.audit_id.replace("M5T_", "");
    b.title = `${r.audit_id}: ${r.human_label || "unlabelled"}`;
    b.onclick = () => { idx = i; renderRow(); };
    g.appendChild(b);
  });
}

function renderRow() {
  const r = rows[idx];
  pendingLabel = r.human_label || "";
  pendingCat = r.error_category || (pendingLabel === "supported" ? "none" : "");
  document.getElementById("posLabel").textContent = `${r.audit_id}  (${idx + 1} / ${rows.length})`;
  document.getElementById("claimSpan").textContent = `CLAIM: "${r.claim_span}"`;
  document.getElementById("claimMeta").innerHTML = `
    <span>Full caption: <em>"${r.caption}"</em></span>
    <span class="tag">object: ${r.object}</span>
    <span class="tag">action: ${r.action}</span>
    ${r.quantity ? `<span class="tag">qty: ${r.quantity}</span>` : ""}
  `;
  document.getElementById("stripImg").src = `/strips/${r.audit_id}.png`;
  document.getElementById("notesInput").value = r.notes || "";
  syncButtons();
  renderNav();
}

function syncButtons() {
  for (const lbl of ["supported", "contradicted", "unverifiable"]) {
    document.getElementById("lbl_" + lbl).classList.toggle("selected", pendingLabel === lbl);
  }
  const cw = document.getElementById("catWrap");
  if (pendingLabel === "contradicted" || pendingLabel === "unverifiable") {
    cw.style.display = "block";
    const cg = document.getElementById("catsGrid");
    cg.innerHTML = "";
    for (const [key, title] of CAT_DEFS) {
      const b = document.createElement("button");
      b.className = "cat-btn" + (pendingCat === key ? " active" : "");
      b.textContent = title;
      b.onclick = () => {
        pendingCat = key;
        syncButtons();
        saveCurrent(document.getElementById("autoAdv").checked);
      };
      cg.appendChild(b);
    }
  } else {
    cw.style.display = "none";
  }
}

function pickLabel(lbl) {
  pendingLabel = lbl;
  if (lbl === "supported") {
    pendingCat = "none";
    syncButtons();
    saveCurrent(document.getElementById("autoAdv").checked);
  } else {
    if (pendingCat === "none") pendingCat = "";
    syncButtons();
  }
}

async function saveCurrent(advance) {
  if (!pendingLabel) { showToast("Pick Supported, Contradicted, or Unverifiable first"); return; }
  if (pendingLabel !== "supported" && (!pendingCat || pendingCat === "none")) {
    showToast("Pick an error category below"); return;
  }
  const r = rows[idx];
  const notes = document.getElementById("notesInput").value.trim();
  const res = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      rater: curRater, audit_id: r.audit_id,
      human_label: pendingLabel,
      error_category: pendingLabel === "supported" ? "none" : pendingCat,
      notes,
    }),
  });
  if (!res.ok) { showToast("Save error!"); return; }
  r.human_label = pendingLabel;
  r.error_category = pendingLabel === "supported" ? "none" : pendingCat;
  r.notes = notes;
  showToast(`Saved ${r.audit_id} (${pendingLabel}) ✓`);
  if (advance && idx < rows.length - 1) {
    idx += 1;
  }
  renderRow();
}

function step(delta) {
  idx = Math.max(0, Math.min(rows.length - 1, idx + delta));
  renderRow();
}

function jumpNextUnlabelled() {
  for (let off = 1; off <= rows.length; off++) {
    const j = (idx + off) % rows.length;
    if (!rows[j].human_label) { idx = j; renderRow(); return; }
  }
  showToast("All 60 claims rated! Thank you!");
}

document.addEventListener("keydown", (e) => {
  if (!curRater) return;
  if (document.activeElement && document.activeElement.tagName === "INPUT") return;
  if (e.key === "1") pickLabel("supported");
  else if (e.key === "2") pickLabel("contradicted");
  else if (e.key === "3") pickLabel("unverifiable");
  else if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowRight") step(1);
});

loadStatus();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/api/status":
            with LOCK:
                out = []
                for rid, meta in RATERS.items():
                    rs = load_rater_rows(rid)
                    done = sum(1 for r in rs if r.get("human_label") in ALLOWED_LABELS)
                    out.append({"id": rid, "name": meta["name"], "slices": meta["slices"], "done": done, "total": len(rs)})
            self._send_json({"raters": out})
            return

        if path == "/api/rows":
            rater = (qs.get("rater") or [""])[0].upper()
            if rater not in RATERS:
                self._send_json({"error": "unknown rater"}, 400)
                return
            with LOCK:
                rs = load_rater_rows(rater)
            self._send_json(rs)
            return

        if path == "/api/download":
            rater = (qs.get("rater") or [""])[0].upper()
            if rater not in RATERS:
                self.send_error(400, "unknown rater")
                return
            with LOCK:
                p = _rater_csv_path(rater)
                data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", f'attachment; filename="rater_{rater}.csv"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/strips/"):
            fname = Path(path).name
            if not (fname.startswith("M5T_") and fname.endswith(".png")):
                self.send_error(404)
                return
            img_p = DATA_DIR / "blind_audit_strips" / fname
            if not img_p.is_file():
                self.send_error(404)
                return
            data = img_p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path != "/api/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(n).decode("utf-8"))
        rater = str(payload.get("rater", "")).upper()
        aid = str(payload.get("audit_id", "")).strip()
        lbl = str(payload.get("human_label", "")).strip().lower()
        cat = str(payload.get("error_category", "none")).strip().lower() or "none"
        notes = str(payload.get("notes", "")).strip()

        if rater not in RATERS or lbl not in ALLOWED_LABELS or cat not in ALLOWED_CATS:
            self._send_json({"error": "invalid payload"}, 400)
            return
        if lbl == "supported":
            cat = "none"

        with LOCK:
            rs = load_rater_rows(rater)
            found = False
            for r in rs:
                if r["audit_id"] == aid:
                    r["human_label"] = lbl
                    r["error_category"] = cat
                    r["notes"] = notes
                    found = True
                    break
            if not found:
                self._send_json({"error": "audit_id not in rater slice"}, 404)
                return
            save_rater_rows(rater, rs)
        self._send_json({"ok": True, "audit_id": aid, "human_label": lbl, "error_category": cat})


def main() -> None:
    global DATA_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    DATA_DIR = Path(args.data_dir).resolve()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TerraWatch Blind Human Audit Web App listening on http://{args.host}:{args.port} (data={DATA_DIR})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
