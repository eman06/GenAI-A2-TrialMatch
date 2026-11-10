"""TrialMatch-Lite web application (Task 6).

Start:  double-click start_app.bat   (or)   python -m app.server
Then open http://127.0.0.1:8000  - everything else happens in the browser.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from trialmatch.attention import reference_attention, sdpa_attention, tiled_attention
from trialmatch.config import ADAPTER_DIR, DRAFT_ADAPTER_DIR, RESULTS, ROOT
from trialmatch.data import load_split, split_free_text
from trialmatch.pipeline import Screener, ScreenerConfig

STATIC = Path(__file__).parent / "static"
REPORTS = RESULTS / "app_reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DISCLAIMER = ("EDUCATIONAL SCREENING PROTOTYPE - NOT A CLINICAL ENROLLMENT DECISION. Outputs come from a small "
              "research model, can be wrong, and must be reviewed by qualified clinical research staff.")

app = FastAPI(title="TrialMatch-Lite")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

STATE = {"status": "loading", "error": None, "screener": None, "loaded_at": None}
LOCK = threading.Lock()   # one CPU model -> serialise inference


def _load():
    try:
        scr = Screener(ScreenerConfig(adapter=str(ADAPTER_DIR), draft_adapter=str(DRAFT_ADAPTER_DIR)))
        STATE.update(screener=scr, status="ready", loaded_at=datetime.now(timezone.utc).isoformat())
    except Exception as e:  # noqa: BLE001
        STATE.update(status="error", error=f"{type(e).__name__}: {e}")
        traceback.print_exc()


threading.Thread(target=_load, daemon=True).start()


def _screener() -> Screener:
    if STATE["status"] == "loading":
        raise HTTPException(503, "The model is still loading (about 20 s after start-up). Please retry shortly.")
    if STATE["status"] == "error":
        raise HTTPException(500, f"Model failed to load: {STATE['error']}. See README 'Troubleshooting'.")
    return STATE["screener"]


@app.exception_handler(RequestValidationError)
async def _validation(_: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        loc = ".".join(str(x) for x in e["loc"][1:]) or "request"
        msgs.append(f"{loc}: {e['msg'].removeprefix('Value error, ')}")
    return JSONResponse({"detail": "; ".join(msgs)}, status_code=422)


# --------------------------------------------------------------------------- schemas
Mode = Literal["kv", "nocache", "tiled", "speculative"]


class Criterion(BaseModel):
    type: Literal["inclusion", "exclusion"]
    text: str = Field(min_length=3, max_length=1500)

    @field_validator("text")
    @classmethod
    def _strip(cls, v):
        v = v.strip()
        if len(v) < 3:
            raise ValueError("criterion text is too short")
        return v


class ScreenRequest(BaseModel):
    note: str = Field(max_length=8000)
    criteria: list[Criterion] = Field(min_length=1, max_length=40)
    mode: Mode = "kv"
    k: int = Field(4, ge=1, le=8)
    trial_id: str | None = Field(None, max_length=40)

    @field_validator("note")
    @classmethod
    def _note(cls, v):
        v = v.strip()
        if len(v) < 20:
            raise ValueError("patient note must contain at least one full sentence (20+ characters)")
        if len(re.findall(r"[A-Za-z]", v)) < 10:
            raise ValueError("patient note does not look like text")
        return v


class CompareRequest(BaseModel):
    note: str = Field(min_length=20, max_length=8000)
    criterion: Criterion
    modes: list[Mode] = ["nocache", "kv", "tiled", "speculative"]
    k: int = Field(4, ge=1, le=8)


class AttnRequest(BaseModel):
    seq_len: int = Field(256, ge=1, le=2048)
    query_len: int | None = Field(None, ge=1, le=2048)
    block: int = Field(64, ge=8, le=512)


class TextRequest(BaseModel):
    text: str = Field(min_length=5, max_length=20000)


# --------------------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    scr = STATE["screener"]
    info = {"status": STATE["status"], "error": STATE["error"], "loaded_at": STATE["loaded_at"],
            "disclaimer": DISCLAIMER, "torch": torch.__version__, "threads": torch.get_num_threads(),
            "device": "cpu"}
    if scr:
        info.update(target=scr.cfg.target, adapter=scr.adapter_used, draft=scr.cfg.draft,
                    draft_adapter=getattr(scr, "draft_adapter_used", None), adapter_sha=_adapter_sha(scr.adapter_used))
    return info


def _adapter_sha(path):
    if not path:
        return None
    p = Path(path, "adapter.pt")
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


@app.post("/api/screen")
def screen(req: ScreenRequest):
    scr = _screener()
    with LOCK:
        try:
            out = scr.screen(req.note, [(c.type, c.text) for c in req.criteria], mode=req.mode, k=req.k)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    rid = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha1(req.note.encode()).hexdigest()[:6]
    out.update(id=rid, trial_id=req.trial_id, created=datetime.now(timezone.utc).isoformat(), disclaimer=DISCLAIMER,
               provenance={"note_sha1": hashlib.sha1(req.note.encode()).hexdigest(),
                           "adapter_sha256_12": _adapter_sha(scr.adapter_used), "app_version": "1.0",
                           "split_note": "model trained on TrialGPT train split (37 patients) only"},
               note=req.note)
    json.dump(out, open(REPORTS / f"{rid}.json", "w"), indent=2)
    return out


@app.post("/api/compare")
def compare(req: CompareRequest):
    """Run ONE criterion through each decoding mode and compare tokens + timings."""
    scr = _screener()
    sents = split_free_text(req.note)
    rows, ref = [], None
    with LOCK:
        for m in req.modes:
            r = scr.screen_criterion(sents, req.criterion.type, req.criterion.text, m, None, req.k)
            toks = r["raw_output"]
            if ref is None:
                ref = toks
            rows.append({"mode": m, "label": r["label"], "evidence": r["evidence"], "output": toks,
                         "same_as_first": toks == ref, **r["timing"]})
    return {"criterion": req.criterion.text, "rows": rows,
            "note": "first listed mode is the reference for output equality; all modes use deterministic greedy decoding"}


@app.post("/api/attention_bench")
def attention_bench(req: AttnRequest):
    """Live micro-benchmark of the three attention implementations on random tensors."""
    S = req.seq_len
    T = min(req.query_len or S, S)
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 15, T, 64), torch.randn(1, 15, S, 64), torch.randn(1, 15, S, 64)
    gold = reference_attention(q.double(), k.double(), v.double())
    impls = {"reference (explicit softmax)": lambda: reference_attention(q, k, v),
             f"tiled online-softmax (block {req.block})": lambda: tiled_attention(q, k, v, block_q=req.block, block_k=req.block),
             "PyTorch SDPA (fused)": lambda: sdpa_attention(q, k, v)}
    rows = []
    with LOCK:
        for name, fn in impls.items():
            o = fn()
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            err = (o.double() - gold).abs()
            rows.append({"impl": name, "median_ms": 1000 * sorted(ts)[1], "max_abs_err": err.max().item(),
                         "mean_abs_err": err.mean().item()})
    return {"T": T, "S": S, "heads": 15, "head_dim": 64, "rows": rows,
            "score_matrix_mb": 15 * T * S * 4 / 2**20, "tile_mb": 15 * min(req.block, T) * min(req.block, S) * 4 / 2**20}


@app.get("/api/examples")
def examples():
    """Validation-split patients (never test) + one hand-written unseen case."""
    df = load_split()["val"]
    out = []
    for pid, sub in list(df.groupby("patient_id"))[:4]:
        trial = sub.trial_id.iloc[0]
        t = sub[sub.trial_id == trial]
        out.append({"name": f"{pid} / {trial} (validation split)", "note": t.note.iloc[0], "trial_id": trial,
                    "trial_title": t.trial_title.iloc[0],
                    "criteria": [{"type": r.criterion_type, "text": r.criterion_text} for _, r in t.iterrows()][:12]})
    out.append({"name": "Unseen hand-written case (not in dataset)", "trial_id": None, "trial_title": "Hypothetical T2DM trial",
                "note": ("A 62-year-old man is referred for poorly controlled type 2 diabetes. His HbA1c is 9.1% despite metformin "
                         "1000 mg twice daily for three years. He has hypertension treated with lisinopril. His eGFR is 48 "
                         "mL/min/1.73m2. He had a myocardial infarction two years ago and had a stent placed. "
                         "He does not smoke and drinks alcohol occasionally. BMI is 33 kg/m2."),
                "criteria": [{"type": "inclusion", "text": "Adults aged 40 to 75 years with type 2 diabetes"},
                             {"type": "inclusion", "text": "HbA1c between 7.5% and 10.5% on stable metformin"},
                             {"type": "inclusion", "text": "Body mass index of 25 kg/m2 or higher"},
                             {"type": "exclusion", "text": "eGFR below 30 mL/min/1.73m2"},
                             {"type": "exclusion", "text": "Current pregnancy or breastfeeding"},
                             {"type": "exclusion", "text": "History of pancreatitis"},
                             {"type": "exclusion", "text": "Myocardial infarction within the past 6 months"}]})
    return out


def split_eligibility_text(text: str) -> list[dict]:
    """Heuristic splitter for a pasted eligibility section (ClinicalTrials.gov style)."""
    crit, cur = [], None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower().rstrip(":")
        if re.match(r"^(key\s+)?inclusion( criteria)?", low):
            cur = "inclusion"; continue
        if re.match(r"^(key\s+)?exclusion( criteria)?", low):
            cur = "exclusion"; continue
        s = re.sub(r"^([-*•]|\d+[.)])\s*", "", s).strip()
        if cur and len(s) >= 3:
            crit.append({"type": cur, "text": s})
    return crit


@app.post("/api/parse_criteria")
def parse_criteria(req: TextRequest):
    crit = split_eligibility_text(req.text)
    if not crit:
        raise HTTPException(400, "No criteria found. Use headings 'Inclusion Criteria:' and 'Exclusion Criteria:' "
                                 "followed by one criterion per line.")
    return {"criteria": crit}


@app.get("/api/trial/{nct_id}")
def fetch_trial(nct_id: str):
    """Public ClinicalTrials.gov metadata - used only to fill the criteria box."""
    nct_id = nct_id.strip().upper()
    if not re.fullmatch(r"NCT\d{8}", nct_id):
        raise HTTPException(400, "Trial id must look like NCT01234567")
    url = (f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
           "?fields=protocolSection.identificationModule,protocolSection.eligibilityModule")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "TrialMatch-Lite/1.0"}), timeout=10) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        raise HTTPException(404 if e.code == 404 else 502, f"ClinicalTrials.gov returned HTTP {e.code} for {nct_id}") from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Could not reach ClinicalTrials.gov ({type(e).__name__}). Paste the criteria manually.") from e
    ps = d.get("protocolSection", {})
    text = ps.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    return {"nct_id": nct_id, "title": ps.get("identificationModule", {}).get("briefTitle", ""),
            "criteria": split_eligibility_text(text), "raw": text}


# --------------------------------------------------------------------------- results & reports
@app.get("/api/experiments")
def experiments():
    """Experiment logs written by the scripts (live files, not screenshots)."""
    out = {}
    for name in ["bench_kv", "bench_spec", "bench_attention"]:
        p = RESULTS / f"{name}.json"
        out[name] = json.load(open(p)) if p.exists() else None
    ev = {}
    for p in sorted((RESULTS / "eval").glob("metrics_*.json")):
        ev[p.stem.removeprefix("metrics_")] = json.load(open(p))
    out["eval"] = ev
    curves = {}
    for p in RESULTS.glob("train_log_*.jsonl"):
        curves[p.stem.removeprefix("train_log_")] = [json.loads(l) for l in open(p) if l.strip()]
    out["train_curves"] = curves
    return out


@app.get("/api/history")
def history():
    items = []
    for p in sorted(REPORTS.glob("*.json"), reverse=True)[:50]:
        try:
            d = json.load(open(p))
            items.append({"id": d["id"], "created": d["created"], "verdict": d["summary"]["verdict"],
                          "n_criteria": len(d["results"]), "mode": d["mode"], "trial_id": d.get("trial_id")})
        except Exception:  # noqa: BLE001
            continue
    return items


@app.get("/api/report/{rid}.json")
def report_json(rid: str):
    p = _report_path(rid)
    return FileResponse(p, filename=f"screening_{rid}.json")


@app.get("/api/report/{rid}.html", response_class=HTMLResponse)
def report_html(rid: str, download: int = 0):
    d = json.load(open(_report_path(rid)))
    page = render_report(d)
    headers = {"Content-Disposition": f'attachment; filename="screening_{rid}.html"'} if download else {}
    return HTMLResponse(page, headers=headers)


def _report_path(rid: str) -> Path:
    if not re.fullmatch(r"[\w-]+", rid):
        raise HTTPException(400, "bad report id")
    p = REPORTS / f"{rid}.json"
    if not p.exists():
        raise HTTPException(404, "report not found")
    return p


def render_report(d: dict) -> str:
    e = html.escape
    s = d["summary"]
    rows = []
    for i, r in enumerate(d["results"], 1):
        ev = "<br>".join(f"<b>[S{x['id']}]</b> {e(x['text'])}" for x in r["evidence_text"]) or "<i>no sentence cited</i>"
        rows.append(f"<tr><td>{i}</td><td>{e(r['criterion_type'])}</td><td>{e(r['criterion'])}</td>"
                    f"<td><b>{e(r['label'])}</b></td><td>{ev}</td><td>{e(r['explanation'])}</td></tr>")
    sents = "".join(f"<li value='{x['id']}'>{e(x['text'])}</li>" for x in d["sentences"])
    lists = "".join(f"<p><b>{e(k.replace('_', ' ').title())}:</b> {e('; '.join(s[k])) if s[k] else 'none'}</p>"
                    for k in ("blocking", "missing_information", "needs_verification"))
    m, pv = d["model"], d["provenance"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Screening report {e(d['id'])}</title>
<style>body{{font-family:system-ui,Segoe UI,Arial;margin:24px;color:#111;max-width:1100px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #bbb;padding:6px;vertical-align:top;font-size:13px}}th{{background:#eef}}
.warn{{background:#fff3cd;border:1px solid #e0b400;padding:10px;font-weight:600}}.v{{font-size:20px;font-weight:700}}</style></head><body>
<div class="warn">{e(d['disclaimer'])}</div>
<h1>Eligibility screening report</h1>
<p>Report id {e(d['id'])} &middot; created {e(d['created'])} &middot; trial {e(str(d.get('trial_id') or 'n/a'))}</p>
<p class="v">Preliminary outcome: {e(s['verdict'])}</p><p>Aggregation rule applied: {e(s['rule'])}</p>{lists}
<h2>Criterion-level decisions</h2><table><tr><th>#</th><th>Type</th><th>Criterion</th><th>Label</th><th>Evidence</th><th>Explanation</th></tr>{''.join(rows)}</table>
<h2>Patient record (sentence ids)</h2><ol start="0">{sents}</ol>
<h2>Provenance</h2><p>Model {e(m['target'])}, adapter {e(str(m['adapter']))} (sha256 {e(str(pv['adapter_sha256_12']))}),
decoding mode {e(d['mode'])} ({e(m['decoding'])}), draft {e(str(m['draft']))}. Note sha1 {e(pv['note_sha1'])}. {e(pv['split_note'])}.
Total inference time {d['total_s']} s.</p>
<p><i>The patient-trial outcome is a rule-based aggregation of criterion-level predictions, not a ground-truth label.</i></p></body></html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, log_level="info")
