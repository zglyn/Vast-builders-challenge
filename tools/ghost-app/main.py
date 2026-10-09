"""GHOST — counterfactual traffic intelligence demo (VAST Builders Challenge).

Secrets stay server-side. Routes are rooted at / because Ingress strips /app.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any
from urllib.parse import quote

import requests
from flask import Flask, Response, jsonify, request, stream_with_context

from ensemble import run_ensemble
from ghost_engine import analyze_detections
from scenario_schema import SUPPORTED_INTERVENTIONS, ScenarioValidationError, validate_scenario
from whatif_parser import parse_whatif

app = Flask(__name__)

VSS_URL = os.environ.get("VSS_URL", "").rstrip("/")
VSS_USERNAME = os.environ.get("VSS_USERNAME", "")
VSS_PASSWORD = os.environ.get("VSS_PASSWORD", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_TEAM = os.environ.get("WANDB_TEAM", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")

# Curated I-24 segment with dense interacting traffic (real indexed footage).
DEMO_SOURCE = os.environ.get(
    "GHOST_DEMO_SOURCE",
    "s3://team-42-vss-chunks-segments/segments/"
    "20261001_055846_scene1_p1c2_chunk_0004_segment_002_of_006.mp4",
)

_token: str | None = None
_token_lock = threading.Lock()
_demo_cache: dict[str, Any] | None = None
_demo_lock = threading.Lock()


def _login(force: bool = False) -> str:
    global _token
    with _token_lock:
        if _token and not force:
            return _token
        r = requests.post(
            f"{VSS_URL}/api/v1/auth/login",
            json={"username": VSS_USERNAME, "password": VSS_PASSWORD},
            timeout=30,
        )
        r.raise_for_status()
        _token = r.json()["access_token"]
        return _token


def _vss(method: str, path: str, *, params=None, json_body=None, stream=False, timeout=120):
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}
    url = f"{VSS_URL}{path}"
    r = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        json=json_body,
        stream=stream,
        timeout=timeout,
    )
    if r.status_code == 401:
        token = _login(force=True)
        headers["Authorization"] = f"Bearer {token}"
        r = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json_body,
            stream=stream,
            timeout=timeout,
        )
    return r


WANDB_MODEL = os.environ.get(
    "WANDB_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)


def _wandb_chat(system: str, user: str, *, max_tokens: int = 350, temperature: float = 0.2) -> dict[str, Any]:
    """Shared W&B Serverless Inference chat helper (credentials stay server-side)."""
    if not WANDB_API_KEY:
        return {"ok": False, "error": "WANDB_API_KEY not set", "text": None}
    body = {
        "model": WANDB_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {WANDB_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Project": f"{WANDB_TEAM}/{WANDB_PROJECT}".strip("/"),
    }
    endpoints = [
        "https://api.inference.wandb.ai/v1/chat/completions",
        "https://api.wandb.ai/proxy/weave/inference/v1/chat/completions",
    ]
    errors = []
    for url in endpoints:
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            if r.status_code >= 400:
                errors.append(f"{url}: HTTP {r.status_code}")
                continue
            text = r.json()["choices"][0]["message"]["content"]
            return {"ok": True, "provider": "wandb", "text": text, "model": WANDB_MODEL}
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
    return {"ok": False, "error": "; ".join(errors)[:400], "text": None}


def _wandb_explain(prompt: str) -> dict[str, Any]:
    """Attempt W&B Serverless Inference; return status if unavailable."""
    return _wandb_chat(
        (
            "You are GHOST, a traffic near-miss analyst. Be concise. "
            "Clearly separate observed facts from image-space estimates. "
            "State uncertainty. Never invent meters or crash outcomes."
        ),
        prompt,
        max_tokens=350,
        temperature=0.2,
    )


def _vss_explain(event: dict, caption: str, similar_blurb: str) -> str:
    """Fallback explanation using VSS search synthesis + structured template."""
    prompt_q = (
        "highway vehicles nearly colliding or braking hard while close together"
    )
    syn = ""
    try:
        r = _vss(
            "POST",
            "/api/v1/search",
            json_body={
                "query": prompt_q,
                "top_k": 5,
                "llm_top_n": 2,
                "min_similarity": 0.15,
                "time_filter": "all",
                "metadata_filters": {"camera_id": "i24_cam-1"},
            },
        )
        if r.ok:
            syn = ((r.json().get("llm_synthesis") or {}).get("response") or "").strip()
    except Exception:  # noqa: BLE001
        syn = ""

    ttc = event.get("ttc_sec_image_space")
    lines = [
        "GHOST analysis (structured; image-space geometry)",
        f"- Event kind: {event.get('kind')}",
        f"- Tracks: {event.get('id_a')} ({event.get('label_a')})"
        + (f" vs {event.get('id_b')} ({event.get('label_b')})" if event.get("id_b") else ""),
        f"- Time in clip: t≈{event.get('t')}s",
        f"- Normalized separation: {event.get('norm_dist')} vehicle-lengths (approx, image)",
        f"- Image-space TTC: {ttc if ttc is not None else 'n/a'} s "
        "(pixels÷closing px/s — not real-world seconds of crash risk)",
        f"- Cosmos caption evidence: {caption[:280]}",
        "",
        "Uncertainty: YOLO boxes jitter; tracker IDs can swap; motion is 2D image projection; "
        "ghost path assumes constant velocity with no brake/swerve.",
    ]
    if syn:
        lines += ["", "Archive synthesis (VSS LLM):", syn[:500]]
    if similar_blurb:
        lines += ["", similar_blurb]
    return "\n".join(lines)


def _build_demo() -> dict[str, Any]:
    meta_r = _vss("GET", "/api/v1/videos/metadata", params={"source": DEMO_SOURCE})
    meta_r.raise_for_status()
    meta = meta_r.json()

    det_r = _vss("GET", "/api/v1/videos/detections", params={"source": DEMO_SOURCE})
    det_r.raise_for_status()
    det = det_r.json()

    analysis = analyze_detections(det)
    primary = analysis.get("primary_event") or {}

    # Similar interactions via VSS search
    search_r = _vss(
        "POST",
        "/api/v1/search",
        json_body={
            "query": "vehicles close together on highway traffic",
            "top_k": 8,
            "llm_top_n": 2,
            "min_similarity": 0.18,
            "time_filter": "all",
            "metadata_filters": {"capture_type": "traffic"},
        },
    )
    similar = []
    search_syn = ""
    if search_r.ok:
        sj = search_r.json()
        search_syn = ((sj.get("llm_synthesis") or {}).get("response") or "").strip()
        for hit in sj.get("results") or []:
            src = hit.get("source")
            if not src or src == DEMO_SOURCE:
                continue
            similar.append(
                {
                    "source": src,
                    "score": hit.get("similarity_score"),
                    "camera_id": hit.get("camera_id"),
                    "location": hit.get("location"),
                    "t_start": hit.get("segment_start_sec"),
                    "t_end": hit.get("segment_end_sec"),
                    "caption": (hit.get("reasoning_content") or "")[:260],
                    "original_video": hit.get("original_video"),
                }
            )
            if len(similar) >= 5:
                break

    caption = meta.get("reasoning_content") or ""
    similar_blurb = f"Similar archive hits: {len(similar)} traffic clips retrieved."
    wandb = _wandb_explain(
        f"Primary event JSON:\n{json.dumps(primary, indent=2)}\n\n"
        f"Cosmos caption:\n{caption}\n\n"
        f"Explain the near-miss candidate and uncertainty in under 180 words."
    )
    if wandb.get("ok"):
        explanation = wandb["text"]
        explanation_source = "wandb"
    else:
        explanation = _vss_explain(primary, caption, similar_blurb)
        explanation_source = "vss_fallback"
        if wandb.get("error"):
            explanation += f"\n\n[W&B inference unavailable: {wandb['error'][:180]}]"

    return {
        "demo_source": DEMO_SOURCE,
        "metadata": {
            "camera_id": meta.get("camera_id"),
            "location": meta.get("location"),
            "capture_type": meta.get("capture_type"),
            "duration": meta.get("duration"),
            "segment_start_sec": meta.get("segment_start_sec"),
            "segment_end_sec": meta.get("segment_end_sec"),
            "object_counts": meta.get("object_counts"),
            "object_classes": meta.get("object_classes"),
            "reasoning_content": caption,
            "original_video": meta.get("original_video"),
        },
        "analysis": analysis,
        "similar": similar,
        "search_synthesis": search_syn,
        "explanation": explanation,
        "explanation_source": explanation_source,
        "generated_at": time.time(),
    }


def get_demo(force: bool = False) -> dict[str, Any]:
    global _demo_cache
    with _demo_lock:
        if _demo_cache and not force:
            return _demo_cache
        _demo_cache = _build_demo()
        return _demo_cache


@app.get("/health")
def health():
    ok = bool(VSS_URL and VSS_USERNAME and VSS_PASSWORD)
    return jsonify({"status": "ok" if ok else "misconfigured", "app": "ghost"})


@app.get("/api/demo")
def api_demo():
    force = request.args.get("refresh") == "1"
    try:
        return jsonify(get_demo(force=force))
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 500


@app.get("/api/video")
def api_video():
    """Proxy VSS stream so the JWT never reaches the browser."""
    source = request.args.get("source") or DEMO_SOURCE
    token = _login()
    upstream = (
        f"{VSS_URL}/api/v1/videos/stream?source={quote(source, safe='')}&token={quote(token)}"
    )
    headers = {}
    if request.headers.get("Range"):
        headers["Range"] = request.headers["Range"]
    r = requests.get(upstream, headers=headers, stream=True, timeout=120)
    excluded = {"content-encoding", "transfer-encoding", "connection"}
    out_headers = {
        k: v for k, v in r.headers.items() if k.lower() not in excluded
    }
    out_headers.setdefault("Content-Type", "video/mp4")
    out_headers["Accept-Ranges"] = "bytes"

    def generate():
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                yield chunk

    return Response(
        stream_with_context(generate()),
        status=r.status_code,
        headers=out_headers,
        direct_passthrough=True,
    )


@app.post("/api/analyze")
def api_analyze():
    body = request.get_json(force=True, silent=True) or {}
    source = body.get("source")
    if not source:
        return jsonify({"error": "source required"}), 400
    det_r = _vss("GET", "/api/v1/videos/detections", params={"source": source})
    if not det_r.ok:
        return jsonify({"error": det_r.text[:300]}), det_r.status_code
    meta_r = _vss("GET", "/api/v1/videos/metadata", params={"source": source})
    meta = meta_r.json() if meta_r.ok else {}
    analysis = analyze_detections(det_r.json())
    return jsonify({"source": source, "metadata": meta, "analysis": analysis})


@app.get("/api/whatif/interventions")
def api_whatif_interventions():
    return jsonify({"interventions": list(SUPPORTED_INTERVENTIONS)})


@app.post("/api/whatif/parse")
def api_whatif_parse():
    """Parse free-form what-if text → schema-validated scenario + editable chips."""
    body = request.get_json(force=True, silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    event = body.get("event") or {}
    analysis = body.get("analysis")
    if analysis is None:
        try:
            analysis = get_demo().get("analysis") or {}
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "message": f"demo unavailable: {e}"}), 500
    known = {int(t["id"]) for t in (analysis.get("tracks") or [])}
    if not event:
        event = analysis.get("primary_event") or {}
    use_llm = body.get("use_llm", True)

    def llm(system: str, user: str) -> dict[str, Any]:
        return _wandb_chat(system, user, max_tokens=280, temperature=0.1)

    result = parse_whatif(
        prompt,
        event=event,
        known_track_ids=known,
        llm_chat=llm if use_llm else None,
    )
    status = 200 if result.get("ok") or result.get("unsupported") else 400
    return jsonify(result), status


@app.post("/api/whatif/simulate")
def api_whatif_simulate():
    """Run multi-agent ensemble through the deterministic ghost engine."""
    body = request.get_json(force=True, silent=True) or {}
    scenario_raw = body.get("scenario") or {}
    event = body.get("event") or {}
    analysis = body.get("analysis")
    if analysis is None:
        try:
            analysis = get_demo().get("analysis") or {}
        except Exception as e:  # noqa: BLE001
            return jsonify({"ok": False, "message": f"demo unavailable: {e}"}), 500
    if not event:
        event = analysis.get("primary_event") or {}
    if not event:
        return jsonify({"ok": False, "message": "No event selected / available."}), 400

    try:
        scenario = validate_scenario(
            scenario_raw,
            known_track_ids={int(t["id"]) for t in (analysis.get("tracks") or [])},
        )
    except ScenarioValidationError as e:
        return jsonify(
            {
                "ok": False,
                "unsupported": e.unsupported,
                "message": str(e),
                "supported": list(SUPPORTED_INTERVENTIONS),
            }
        ), 400

    n_runs = int(body.get("n_runs") or 8)
    seed = int(body.get("seed") or 42)
    use_llm = bool(body.get("use_llm", True))
    use_agents = bool(body.get("use_agents", True))
    personas = body.get("personas")

    def llm(system: str, user: str) -> dict[str, Any]:
        return _wandb_chat(system, user, max_tokens=220, temperature=0.2)

    try:
        result = run_ensemble(
            analysis=analysis,
            event=event,
            scenario=scenario,
            n_runs=n_runs,
            seed=seed,
            personas=personas,
            llm_chat=llm if use_llm else None,
            model_name=WANDB_MODEL,
            use_agents=use_agents,
        )
    except Exception as e:  # noqa: BLE001
        return jsonify({"ok": False, "message": str(e)[:400]}), 500
    return jsonify(result)


@app.get("/")
def index():
    return INDEX_HTML


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>GHOST — Counterfactual Traffic Intelligence</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@600;700;800&display=swap" rel="stylesheet"/>
<style>
:root {
  --bg: #0b0e12;
  --panel: #12171e;
  --line: #243041;
  --text: #e8eef7;
  --muted: #8b9bb0;
  --green: #3dffa8;
  --red: #ff5c5c;
  --amber: #ffc14d;
  --blue: #6eb6ff;
  --danger-bg: rgba(255,92,92,0.12);
}
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:var(--bg); color:var(--text);
  font-family:"IBM Plex Mono", ui-monospace, monospace; color-scheme: dark; }
body {
  background:
    radial-gradient(1200px 600px at 10% -10%, #162033 0%, transparent 55%),
    radial-gradient(900px 500px at 100% 0%, #1a1218 0%, transparent 50%),
    var(--bg);
}
.shell { max-width: 1400px; margin: 0 auto; padding: 20px 20px 40px; }
header { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; margin-bottom:18px; }
.brand { font-family: Syne, sans-serif; font-weight:800; font-size: clamp(2rem, 4vw, 3.2rem); letter-spacing:-0.03em; line-height:0.95; }
.brand span { color: var(--red); }
.tag { color:var(--muted); font-size:0.78rem; max-width: 420px; text-align:right; }
.grid { display:grid; grid-template-columns: 1.45fr 0.9fr; gap:16px; }
@media (max-width: 980px){ .grid{ grid-template-columns:1fr; } .tag{ text-align:left; } }
.panel { background: var(--panel); border:1px solid var(--line); border-radius: 4px; }
.stage { position:relative; background:#000; aspect-ratio: 16/9; overflow:hidden; }
video, canvas { position:absolute; inset:0; width:100%; height:100%; }
canvas { pointer-events:none; }
.controls { padding:12px 14px; display:flex; flex-direction:column; gap:10px; border-top:1px solid var(--line); }
.row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
button {
  background:#1a2330; color:var(--text); border:1px solid var(--line);
  padding:8px 12px; border-radius:3px; cursor:pointer; font:inherit; font-size:0.78rem;
}
button:hover { border-color: var(--blue); }
button.primary { background: #1d2b24; border-color:#2f5a45; color: var(--green); }
button.danger { background: var(--danger-bg); border-color:#6a3030; color: var(--red); }
input[type=range] { flex:1; accent-color: var(--amber); }
.legend { display:flex; gap:14px; font-size:0.72rem; color:var(--muted); }
.dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
.dot.g { background:var(--green);} .dot.r { background:var(--red);} .dot.a { background:var(--amber);}
.side { display:flex; flex-direction:column; gap:12px; }
.side .panel { padding:14px; }
h2 { font-family:Syne,sans-serif; font-size:0.95rem; margin:0 0 10px; letter-spacing:0.02em; }
.meta { font-size:0.72rem; color:var(--muted); line-height:1.55; }
.meta b { color:var(--text); font-weight:500; }
.event {
  border:1px solid var(--line); padding:10px; margin-bottom:8px; cursor:pointer; border-radius:3px;
}
.event:hover, .event.active { border-color: var(--red); background: rgba(255,92,92,0.06); }
.event .k { color:var(--amber); font-size:0.68rem; text-transform:uppercase; }
.event .t { font-size:0.78rem; margin-top:4px; }
.similar { font-size:0.72rem; border-top:1px solid var(--line); padding-top:8px; margin-top:8px; }
.similar a { color:var(--blue); text-decoration:none; cursor:pointer; }
.explain { white-space:pre-wrap; font-size:0.72rem; line-height:1.5; color:#c9d4e3; max-height:280px; overflow:auto; }
.caveat { margin-top:10px; font-size:0.68rem; color:#a08b6a; border-top:1px dashed #3a3428; padding-top:8px; }
.status { font-size:0.72rem; color:var(--muted); }
.err { color:var(--red); }
.tabs { display:flex; gap:6px; margin-bottom:10px; flex-wrap:wrap; }
.tab {
  background:transparent; border:1px solid var(--line); color:var(--muted);
  padding:6px 10px; border-radius:3px; cursor:pointer; font:inherit; font-size:0.72rem;
}
.tab.active { color:var(--text); border-color:var(--amber); background:rgba(255,193,77,0.08); }
.lab-block { display:none; }
.lab-block.active { display:block; }
textarea, input[type=text], input[type=number], #whatifPrompt {
  width:100%; background:#0e141c; color:#f4f7fb; -webkit-text-fill-color:#f4f7fb;
  caret-color: var(--amber); border:1px solid var(--line); border-radius:3px;
  padding:10px; font:inherit; font-size:0.78rem; line-height:1.4; resize:vertical;
  color-scheme: dark;
}
textarea::placeholder, input::placeholder, #whatifPrompt::placeholder {
  color:#8b9bb0; -webkit-text-fill-color:#8b9bb0; opacity:1;
}
textarea:focus, input[type=text]:focus, input[type=number]:focus, #whatifPrompt:focus {
  outline:1px solid var(--amber); border-color:var(--amber);
  color:#f4f7fb; -webkit-text-fill-color:#f4f7fb;
}
/* Defeat browser autofill forcing black text on dark fields */
textarea:-webkit-autofill,
textarea:-webkit-autofill:focus,
input:-webkit-autofill,
input:-webkit-autofill:focus,
#whatifPrompt:-webkit-autofill {
  -webkit-text-fill-color:#f4f7fb !important;
  caret-color: var(--amber);
  transition: background-color 99999s ease-in-out 0s;
  box-shadow: 0 0 0 1000px #0e141c inset;
}
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
.chip {
  border:1px solid var(--line); background:#182232; color:var(--text);
  padding:4px 8px; border-radius:3px; font-size:0.68rem; display:flex; gap:6px; align-items:center;
}
.chip b { color:var(--amber); font-weight:500; }
.chip input {
  width:auto; min-width:64px; max-width:140px; padding:2px 4px; font-size:0.68rem;
  background:#0e141c; color:#e8eef7; -webkit-text-fill-color:#e8eef7; border:1px solid var(--line);
}
.driver-card {
  border:1px solid var(--line); padding:8px; margin-bottom:6px; border-radius:3px; cursor:pointer;
  font-size:0.7rem; line-height:1.4;
}
.driver-card:hover, .driver-card.active { border-color:var(--red); background:var(--danger-bg); }
.driver-card .p { color:var(--blue); }
.outcome { font-size:0.72rem; color:#c9d4e3; line-height:1.5; margin:8px 0; }
.hyp-banner {
  font-size:0.68rem; color:var(--red); border:1px dashed #6a3030; padding:8px;
  margin-bottom:10px; background:var(--danger-bg);
}
.progress { font-size:0.68rem; color:var(--muted); max-height:70px; overflow:auto; white-space:pre-wrap; }
.view-toggle { display:flex; gap:6px; }
.view-toggle button.active { border-color:var(--green); color:var(--green); }
.view-toggle button.active.danger { border-color:var(--red); color:var(--red); }
</style>
</head>
<body>
<div class="shell">
  <header>
    <div>
      <div class="brand">GH<span>O</span>ST</div>
      <div class="status" id="status">Loading demo…</div>
    </div>
    <div class="tag">Counterfactual traffic intelligence — reconstruct motion, then ask: what if they didn’t brake?</div>
  </header>

  <div class="grid">
    <div class="panel">
      <div class="stage" id="stage">
        <video id="video" playsinline preload="auto"></video>
        <canvas id="overlay"></canvas>
      </div>
      <div class="controls">
        <div class="row">
          <button class="primary" id="btnPlay">Play</button>
          <button id="btnPause">Pause</button>
          <button class="danger" id="btnFreeze">Freeze near-miss</button>
          <button id="btnReplay">Replay event</button>
          <span class="status" id="clock">t=0.00s</span>
        </div>
        <div class="row">
          <input type="range" id="scrub" min="0" max="1000" value="0"/>
        </div>
        <div class="row">
          <div class="view-toggle">
            <button id="btnReality" class="active">Reality</button>
            <button id="btnHypo" class="danger">Hypothetical</button>
          </div>
          <span class="status" id="viewMode">OBSERVATION</span>
        </div>
        <div class="legend">
          <span><i class="dot g"></i>observed trajectory</span>
          <span><i class="dot r"></i>hypothetical ghost / fan</span>
          <span><i class="dot a"></i>event frame</span>
        </div>
      </div>
    </div>

    <div class="side">
      <div class="panel">
        <div class="tabs">
          <button class="tab active" data-tab="evidence">Evidence</button>
          <button class="tab" data-tab="whatif">What-If Lab</button>
          <button class="tab" data-tab="report">Report</button>
        </div>

        <div class="lab-block active" id="tab-evidence">
          <h2>Clip evidence</h2>
          <div class="meta" id="meta">—</div>
          <h2 style="margin-top:14px">Near-miss candidates</h2>
          <div id="events"></div>
          <h2 style="margin-top:14px">Explanation</h2>
          <div class="explain" id="explain">—</div>
          <div class="caveat" id="caveats"></div>
          <h2 style="margin-top:14px">Similar interactions</h2>
          <div id="similar" class="meta"></div>
        </div>

        <div class="lab-block" id="tab-whatif">
          <div class="hyp-banner">Simulated driver behavior under assumed profiles. Not a prediction or a finding of fault. OBSERVATIONS stay on Reality; ghosts are HYPOTHETICAL.</div>
          <h2>Ask a What If</h2>
          <textarea id="whatifPrompt" rows="3" placeholder="e.g. What if the white truck had braked earlier?"></textarea>
          <div class="row" style="margin-top:8px">
            <button class="primary" id="btnParse">Parse scenario</button>
            <button class="danger" id="btnSimulate" disabled>Run ensemble</button>
            <label class="status"><input type="checkbox" id="useLlm" checked/> use W&amp;B LLM</label>
          </div>
          <div class="chips" id="chips"></div>
          <div class="status" id="parseMsg"></div>
          <div class="progress" id="simProgress"></div>
          <div class="outcome" id="outcome"></div>
          <h2>Driver cards</h2>
          <div id="driverCards" class="meta">Parse + run to see per-agent decisions.</div>
        </div>

        <div class="lab-block" id="tab-report">
          <h2>Ghost Report</h2>
          <div class="hyp-banner">Report mixes OBSERVED clip metadata with HYPOTHETICAL simulation assumptions.</div>
          <div class="explain" id="reportBody">Run a what-if simulation to populate the report.</div>
          <div class="row" style="margin-top:8px">
            <button id="btnPrintReport">Print / Save PDF</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
let demo = null;
let activeEventIdx = 0;
let freezeAt = null;
let viewMode = 'reality'; // reality | hypothetical
let parsedScenario = null;
let simResult = null;
let selectedRunIdx = 0;
let highlightTrackId = null;

function setStatus(msg, isErr=false){
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = 'status' + (isErr ? ' err' : '');
}

document.querySelectorAll('.tab').forEach(tab=>{
  tab.onclick = ()=>{
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.lab-block').forEach(b=>b.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
  };
});

function setViewMode(mode){
  viewMode = mode;
  document.getElementById('btnReality').classList.toggle('active', mode==='reality');
  document.getElementById('btnHypo').classList.toggle('active', mode==='hypothetical');
  document.getElementById('viewMode').textContent = mode==='reality' ? 'OBSERVATION' : 'HYPOTHETICAL';
  document.getElementById('viewMode').style.color = mode==='reality' ? 'var(--green)' : 'var(--red)';
  draw();
}
document.getElementById('btnReality').onclick = ()=> setViewMode('reality');
document.getElementById('btnHypo').onclick = ()=> setViewMode('hypothetical');

async function loadDemo(){
  setStatus('Analyzing I-24 detections + building tracks…');
  const r = await fetch('api/demo');
  if(!r.ok) throw new Error(await r.text());
  demo = await r.json();
  const a = demo.analysis;
  const m = demo.metadata;
  document.getElementById('meta').innerHTML = `
    <div><b>camera</b> ${m.camera_id} · <b>location</b> ${m.location}</div>
    <div><b>duration</b> ${m.duration}s · <b>objects</b> ${JSON.stringify(m.object_counts)}</div>
    <div><b>tracks</b> ${a.n_tracks} (segment-local IDs)</div>
    <div style="margin-top:8px">${(m.reasoning_content||'').slice(0,320)}</div>`;
  document.getElementById('explain').textContent = demo.explanation || '';
  document.getElementById('caveats').textContent = (a.caveats||[]).join(' · ');
  renderEvents();
  renderSimilar();
  video.src = 'api/video?source=' + encodeURIComponent(demo.demo_source);
  await video.play().catch(()=>{});
  video.pause();
  if(a.primary_event){
    freezeAt = a.primary_event.t;
    video.currentTime = Math.max(0, freezeAt - 0.15);
  }
  setStatus(`Ready · ${a.n_tracks} tracks · ${ (a.events||[]).length } events · explain via ${demo.explanation_source}`);
  draw();
}

function renderEvents(){
  const box = document.getElementById('events');
  const events = demo.analysis.events || [];
  box.innerHTML = events.slice(0,8).map((e,i)=>`
    <div class="event ${i===activeEventIdx?'active':''}" data-i="${i}">
      <div class="k">${e.kind}</div>
      <div class="t">t=${Number(e.t).toFixed(2)}s · tracks ${e.id_a}${e.id_b?(' / '+e.id_b):''}
      ${e.ttc_sec_image_space!=null?` · TTC≈${e.ttc_sec_image_space}s (image)`:''}</div>
      <div class="t" style="color:var(--muted)">${e.px_dist!=null?`sep ${e.px_dist}px · norm ${e.norm_dist}`:(e.speed_early_px_s?`speed ${e.speed_early_px_s}→${e.speed_late_px_s} px/s`:'')}</div>
    </div>`).join('') || '<div class="meta">No events found</div>';
  box.querySelectorAll('.event').forEach(el=>{
    el.onclick = ()=>{
      activeEventIdx = Number(el.dataset.i);
      const e = demo.analysis.events[activeEventIdx];
      freezeAt = e.t;
      video.currentTime = Math.max(0, e.t - 0.1);
      video.pause();
      renderEvents();
      // rebuild ghosts for selected event client-side from full paths
      rebuildGhostsForEvent(e);
      draw();
    };
  });
}

function rebuildGhostsForEvent(e){
  const tracks = demo.analysis.tracks || [];
  const fps = demo.analysis.fps || 30;
  const ghosts = [];
  for (const tid of [e.id_a, e.id_b]){
    if(tid==null) continue;
    const tr = tracks.find(t=>t.id===tid);
    if(!tr || !tr.full_path || tr.full_path.length<3) continue;
    const path = tr.full_path;
    let idx = path.findIndex(p=>p.frame===e.frame);
    if(idx<0) idx = path.findIndex(p=>Math.abs(p.t-e.t)<0.05);
    if(idx<0) idx = Math.max(0, path.length-2);
    const look = Math.max(0, idx-5);
    const p0 = path[idx], p1 = path[look];
    const dt = (p0.t - p1.t) || (1/fps);
    const vx = (p0.cx - p1.cx)/dt, vy=(p0.cy-p1.cy)/dt;
    const ghost_path=[];
    for(let i=0;i<=Math.round(1.5*fps);i++){
      const t=i/fps;
      ghost_path.push({t:p0.t+t, frame:p0.frame+i, cx:p0.cx+vx*t, cy:p0.cy+vy*t});
    }
    ghosts.push({track_id:tid, label:tr.label, event_frame:p0.frame, event_t:p0.t,
      ghost_path, actual_path: path.filter(p=>p.frame>=p0.frame).slice(0, ghost_path.length)});
  }
  demo.analysis.ghosts = ghosts;
  demo.analysis.primary_event = e;
}

function renderSimilar(){
  const el = document.getElementById('similar');
  const items = demo.similar || [];
  el.innerHTML = items.map(s=>`
    <div class="similar">
      <div><b>${s.camera_id||'?'}</b> · ${(s.location||'')} · score ${(s.score||0).toFixed(3)}
        · t ${s.t_start??'?'}-${s.t_end??'?'}s</div>
      <div>${(s.caption||'').slice(0,160)}</div>
      <a data-src="${s.source}">analyze this clip</a>
    </div>`).join('') || 'No similar clips';
  el.querySelectorAll('a[data-src]').forEach(a=>{
    a.onclick = async (ev)=>{
      ev.preventDefault();
      setStatus('Analyzing selected clip…');
      const r = await fetch('api/analyze', {method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({source:a.dataset.src})});
      if(!r.ok){ setStatus('Analyze failed', true); return; }
      const data = await r.json();
      demo.demo_source = data.source;
      demo.metadata = {
        camera_id: data.metadata.camera_id,
        location: data.metadata.location,
        duration: data.metadata.duration,
        object_counts: data.metadata.object_counts,
        reasoning_content: data.metadata.reasoning_content,
      };
      demo.analysis = data.analysis;
      activeEventIdx = 0;
      document.getElementById('meta').innerHTML = `
        <div><b>camera</b> ${demo.metadata.camera_id} · <b>location</b> ${demo.metadata.location}</div>
        <div><b>tracks</b> ${data.analysis.n_tracks}</div>
        <div style="margin-top:8px">${(demo.metadata.reasoning_content||'').slice(0,320)}</div>`;
      renderEvents();
      if(data.analysis.primary_event) rebuildGhostsForEvent(data.analysis.primary_event);
      video.src = 'api/video?source=' + encodeURIComponent(data.source);
      video.currentTime = Math.max(0, (data.analysis.primary_event?.t||0) - 0.1);
      setStatus(`Loaded alternate clip · ${data.analysis.n_tracks} tracks`);
      draw();
    };
  });
}

function resize(){
  const stage = document.getElementById('stage');
  canvas.width = stage.clientWidth * devicePixelRatio;
  canvas.height = stage.clientHeight * devicePixelRatio;
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
}

function scale(){
  const [H,W] = demo?.analysis?.video_shape || [1080,1920];
  const sw = canvas.clientWidth / W;
  const sh = canvas.clientHeight / H;
  return {sw, sh, W, H};
}

function draw(){
  if(!demo) return;
  resize();
  const {sw, sh} = scale();
  ctx.clearRect(0,0,canvas.clientWidth, canvas.clientHeight);
  const t = video.currentTime || 0;
  const tracks = demo.analysis.tracks || [];
  const ghosts = demo.analysis.ghosts || [];
  const primary = demo.analysis.primary_event;

  // actual trajectories up to now
  for(const tr of tracks){
    const path = tr.full_path || tr.path || [];
    const pts = path.filter(p=>p.t <= t + 0.01);
    if(pts.length<2) continue;
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(61,255,168,0.55)';
    ctx.lineWidth = 2;
    pts.forEach((p,i)=>{
      const x=p.cx*sw, y=p.cy*sh;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    const last = pts[pts.length-1];
    if(Math.abs(last.t - t) < 0.08 || (path.length && t >= path[0].t && t <= path[path.length-1].t)){
      // nearest bbox
      let best=null, bd=1e9;
      for(const p of path){ const d=Math.abs(p.t-t); if(d<bd){bd=d; best=p;} }
      if(best && bd<0.12){
        const b=best.bbox;
        ctx.strokeStyle = 'rgba(110,182,255,0.9)';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(b[0]*sw, b[1]*sh, (b[2]-b[0])*sw, (b[3]-b[1])*sh);
        ctx.fillStyle = 'rgba(11,14,18,0.75)';
        ctx.fillRect(b[0]*sw, b[1]*sh - 16, 54, 14);
        ctx.fillStyle = '#cfe3ff';
        ctx.font = '11px IBM Plex Mono';
        ctx.fillText(`#${tr.id} ${tr.label}`, b[0]*sw + 3, b[1]*sh - 5);
      }
    }
  }

  // ghost trajectories / fan — only in hypothetical view
  if(viewMode==='hypothetical' && primary && t >= (primary.t - 0.05)){
    // ensemble fan (uncertainty envelope)
    if(simResult && simResult.fan){
      for(const [tid, paths] of Object.entries(simResult.fan)){
        const hi = highlightTrackId!=null && String(highlightTrackId)===String(tid);
        for(const item of paths){
          const gp = item.path || [];
          if(gp.length<2) continue;
          ctx.beginPath();
          ctx.setLineDash([4,6]);
          ctx.strokeStyle = hi ? 'rgba(255,92,92,0.85)' : 'rgba(255,92,92,0.22)';
          ctx.lineWidth = hi ? 2.5 : 1.2;
          gp.forEach((p,i)=>{
            const x=p.cx*sw, y=p.cy*sh;
            if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
          });
          ctx.stroke();
        }
      }
      ctx.setLineDash([]);
    }
    const runGhosts = (simResult && simResult.runs && simResult.runs[selectedRunIdx])
      ? (simResult.runs[selectedRunIdx].ghosts || [])
      : ghosts;
    for(const g of runGhosts){
      const gp = g.ghost_path || [];
      if(gp.length<2) continue;
      const hi = highlightTrackId!=null && g.track_id===highlightTrackId;
      ctx.beginPath();
      ctx.setLineDash([6,5]);
      ctx.strokeStyle = hi ? 'rgba(255,92,92,1)' : 'rgba(255,92,92,0.9)';
      ctx.lineWidth = hi ? 3.2 : 2.5;
      gp.forEach((p,i)=>{
        const x=p.cx*sw, y=p.cy*sh;
        if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      const tip = gp[gp.length-1];
      ctx.fillStyle = 'rgba(255,92,92,0.85)';
      ctx.beginPath();
      ctx.arc(tip.cx*sw, tip.cy*sh, 4, 0, Math.PI*2);
      ctx.fill();
      ctx.fillStyle = '#ffb4b4';
      ctx.font = '11px IBM Plex Mono';
      ctx.fillText(`HYPOTHETICAL #${g.track_id}`, tip.cx*sw + 6, tip.cy*sh - 6);
    }
    // event marker
    ctx.strokeStyle = 'rgba(255,193,77,0.85)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 12); ctx.lineTo(canvas.clientWidth, 12);
    ctx.stroke();
    ctx.fillStyle = '#ffc14d';
    ctx.font = '11px IBM Plex Mono';
    ctx.fillText(`EVENT t=${Number(primary.t).toFixed(2)}s · ${primary.kind} · HYPOTHETICAL`, 8, 11);
  }
}

function renderChips(chips){
  const box = document.getElementById('chips');
  if(!chips || !chips.length){ box.innerHTML = ''; return; }
  box.innerHTML = chips.map(c=>`
    <div class="chip" data-key="${c.key}">
      <b>${c.label}</b>
      <input type="text" value="${String(c.value).replace(/"/g,'&quot;')}" data-key="${c.key}"/>
    </div>`).join('');
}

function chipsToScenario(){
  if(!parsedScenario) return null;
  const sc = JSON.parse(JSON.stringify(parsedScenario));
  document.querySelectorAll('#chips input').forEach(inp=>{
    const key = inp.dataset.key;
    const val = inp.value.trim();
    if(key==='intervention') sc.intervention = val;
    else if(key==='targets') sc.targetTrackIds = val.split(/[,\s]+/).filter(Boolean).map(Number);
    else if(key==='horizon') sc.horizonSec = Number(val)||1.5;
    else if(key.startsWith('param:')){
      const pk = key.slice(6);
      sc.params = sc.params || {};
      sc.params[pk] = Number(val);
      if(Number.isNaN(sc.params[pk])) sc.params[pk] = val;
    } else if(key==='rationale') sc.rationale = val;
  });
  return sc;
}

function renderDriverCards(){
  const el = document.getElementById('driverCards');
  if(!simResult || !simResult.runs || !simResult.runs.length){
    el.textContent = 'Parse + run to see per-agent decisions.';
    return;
  }
  const run = simResult.runs[selectedRunIdx] || simResult.runs[0];
  const decisions = run.decisions || [];
  el.innerHTML = `
    <div class="row" style="margin-bottom:8px">
      <span class="status">run</span>
      <input type="number" id="runPicker" min="0" max="${simResult.runs.length-1}" value="${selectedRunIdx}" style="width:64px"/>
      <span class="status">of ${simResult.runs.length-1} · outcome=${(run.assessment||{}).outcome}</span>
    </div>` +
    (decisions.map(d=>`
      <div class="driver-card" data-tid="${d.track_id}">
        <div><b>#${d.track_id}</b> ${d.label||''} · <span class="p">${d.persona||'?'}</span></div>
        <div>${d.action} · intensity ${Number(d.intensity).toFixed(2)} · delay ${Number(d.reactionDelaySec).toFixed(2)}s</div>
        <div style="color:var(--muted)">${d.reasoning||''}</div>
        <div style="color:var(--amber)">source: ${d.source||'?'} · hazards: ${(d.perceivedHazardIds||[]).join(', ')||'—'}</div>
      </div>`).join('') || '<div class="meta">No agent decisions (intervention-only path).</div>');
  const picker = document.getElementById('runPicker');
  if(picker){
    picker.onchange = ()=>{
      selectedRunIdx = Math.max(0, Math.min(simResult.runs.length-1, Number(picker.value)||0));
      renderDriverCards();
      draw();
    };
  }
  el.querySelectorAll('.driver-card').forEach(card=>{
    card.onclick = ()=>{
      highlightTrackId = Number(card.dataset.tid);
      el.querySelectorAll('.driver-card').forEach(c=>c.classList.remove('active'));
      card.classList.add('active');
      setViewMode('hypothetical');
      draw();
    };
  });
}

function renderReport(){
  const el = document.getElementById('reportBody');
  if(!simResult){ el.textContent = 'Run a what-if simulation to populate the report.'; return; }
  const sc = simResult.scenario || {};
  const sum = simResult.summary || {};
  const run = simResult.runs[selectedRunIdx] || {};
  const lines = [
    'GHOST REPORT',
    '--- OBSERVED ---',
    `Event: ${JSON.stringify(simResult.event)}`,
    `Clip: ${demo?.demo_source||''}`,
    '--- HYPOTHETICAL SCENARIO ---',
    JSON.stringify(sc, null, 2),
    '--- ASSUMPTIONS / PERSONAS ---',
    'Persona params are labeled assumptions from persona_config.py — not real drivers.',
    JSON.stringify(run.persona_assignment || {}, null, 2),
    '--- DRIVER DECISIONS (selected run) ---',
    JSON.stringify(run.decisions || [], null, 2),
    '--- OUTCOME SUMMARY ---',
    sum.outcome_share_text || '',
    sum.disclaimer || '',
    'Image-space geometry only. Not a crash prediction or finding of fault.',
  ];
  el.textContent = lines.join('\n');
}

document.getElementById('btnParse').onclick = async ()=>{
  const prompt = document.getElementById('whatifPrompt').value;
  const event = demo?.analysis?.events?.[activeEventIdx] || demo?.analysis?.primary_event;
  document.getElementById('parseMsg').textContent = 'Parsing…';
  const r = await fetch('api/whatif/parse', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      prompt,
      event,
      analysis: demo?.analysis,
      use_llm: document.getElementById('useLlm').checked,
    })
  });
  const data = await r.json();
  if(!data.ok){
    document.getElementById('parseMsg').innerHTML = `<span class="err">${data.message||'Parse failed'}</span>` +
      (data.supported ? `<div class="status">Supported: ${data.supported.join(', ')}</div>` : '');
    document.getElementById('btnSimulate').disabled = true;
    parsedScenario = null;
    renderChips([]);
    return;
  }
  parsedScenario = data.scenario;
  renderChips(data.chips||[]);
  document.getElementById('parseMsg').textContent = `${data.notice||'OK'} · source=${data.source}`;
  document.getElementById('btnSimulate').disabled = false;
};

document.getElementById('btnSimulate').onclick = async ()=>{
  const scenario = chipsToScenario();
  if(!scenario){ document.getElementById('parseMsg').textContent = 'Parse a scenario first.'; return; }
  const event = demo?.analysis?.events?.[activeEventIdx] || demo?.analysis?.primary_event;
  document.getElementById('simProgress').textContent = 'agents deciding…';
  document.getElementById('outcome').textContent = '';
  setStatus('Running what-if ensemble…');
  const r = await fetch('api/whatif/simulate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({
      scenario,
      event,
      analysis: demo?.analysis,
      n_runs: 8,
      seed: 42,
      use_llm: document.getElementById('useLlm').checked,
      use_agents: true,
    })
  });
  const data = await r.json();
  if(!data.ok){
    document.getElementById('simProgress').innerHTML = `<span class="err">${data.message||'Simulate failed'}</span>`;
    setStatus('Simulate failed', true);
    return;
  }
  simResult = data;
  selectedRunIdx = data.summary?.most_interacting_run ?? 0;
  document.getElementById('simProgress').textContent = (data.progress||[]).join('\n');
  document.getElementById('outcome').textContent = (data.summary?.outcome_share_text||'') + '\n' + (data.notice||'');
  if(data.runs && data.runs[selectedRunIdx]){
    demo.analysis.ghosts = data.runs[selectedRunIdx].ghosts || demo.analysis.ghosts;
  }
  renderDriverCards();
  renderReport();
  setViewMode('hypothetical');
  setStatus(`Ensemble done · cache=${data.cache_hit?'hit':'miss'} · ${data.summary?.outcome_share_text||''}`);
  draw();
};

document.getElementById('btnPrintReport').onclick = ()=> window.print();

video.addEventListener('timeupdate', ()=>{
  document.getElementById('clock').textContent = `t=${video.currentTime.toFixed(2)}s`;
  const dur = video.duration || demo?.metadata?.duration || 5;
  document.getElementById('scrub').value = Math.round(1000 * video.currentTime / dur);
  if(freezeAt!=null && video.currentTime >= freezeAt && !video.paused){
    video.pause();
  }
  draw();
});
video.addEventListener('loadedmetadata', ()=>{ resize(); draw(); });
window.addEventListener('resize', ()=>{ resize(); draw(); });

document.getElementById('btnPlay').onclick = ()=>{ freezeAt=null; video.play(); };
document.getElementById('btnPause').onclick = ()=> video.pause();
document.getElementById('btnFreeze').onclick = ()=>{
  const e = demo?.analysis?.events?.[activeEventIdx] || demo?.analysis?.primary_event;
  if(!e) return;
  freezeAt = e.t;
  video.currentTime = Math.max(0, e.t - 0.05);
  video.pause();
  draw();
};
document.getElementById('btnReplay').onclick = ()=>{
  const e = demo?.analysis?.events?.[activeEventIdx] || demo?.analysis?.primary_event;
  if(!e) return;
  freezeAt = e.t;
  video.currentTime = Math.max(0, e.t - 1.0);
  video.play();
};
document.getElementById('scrub').oninput = (ev)=>{
  const dur = video.duration || demo?.metadata?.duration || 5;
  video.currentTime = (Number(ev.target.value)/1000) * dur;
  draw();
};

loadDemo().catch(err=> setStatus(String(err), true));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    # Warm demo in background so first UI load is faster
    def _warm():
        try:
            get_demo()
            print("demo cache warmed", flush=True)
        except Exception as e:  # noqa: BLE001
            print("demo warm failed:", e, flush=True)

    threading.Thread(target=_warm, daemon=True).start()
    app.run(host="0.0.0.0", port=port, threaded=True)
