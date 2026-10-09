"""GHOST NYC. Credentials stay in the process environment."""

from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import quote

import requests
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from futures import investigation_brief, occlusion_zones, redesign_notes, simulate_futures
from risk_engine import analyze_detections, explain_factors

app = Flask(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))

VSS_URL = os.environ.get("VSS_URL", "").rstrip("/")
VSS_USERNAME = os.environ.get("VSS_USERNAME", "")
VSS_PASSWORD = os.environ.get("VSS_PASSWORD", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_TEAM = os.environ.get("WANDB_TEAM", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "")

DEFAULT_SOURCE = os.environ.get(
    "GHOST_NYC_SOURCE",
    "s3://team-42-vss-chunks-segments/segments/"
    "20261008_074640_GX050001_chunk_0000_segment_001_of_006.mp4",
)
BIKE_CAMERA = "nyc_bike_gopro-1"

_token: str | None = None
_token_lock = threading.Lock()
_clips: dict[str, dict[str, Any]] = {}
_clips_lock = threading.Lock()
_catalog_cache: dict[str, Any] | None = None
_catalog_lock = threading.Lock()


def _login(force: bool = False) -> str:
    global _token
    with _token_lock:
        if _token and not force:
            return _token
        response = requests.post(
            f"{VSS_URL}/api/v1/auth/login",
            json={"username": VSS_USERNAME, "password": VSS_PASSWORD},
            timeout=30,
        )
        response.raise_for_status()
        _token = response.json()["access_token"]
        return _token


def _vss(method: str, path: str, *, params=None, json_body=None, stream=False, timeout=90):
    token = _login()
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.request(
        method,
        f"{VSS_URL}{path}",
        headers=headers,
        params=params,
        json=json_body,
        stream=stream,
        timeout=timeout,
    )
    if response.status_code == 401:
        token = _login(force=True)
        headers["Authorization"] = f"Bearer {token}"
        response = requests.request(
            method,
            f"{VSS_URL}{path}",
            headers=headers,
            params=params,
            json=json_body,
            stream=stream,
            timeout=timeout,
        )
    return response


def _wandb(evidence: str) -> dict[str, Any]:
    if not WANDB_API_KEY:
        return {"ok": False, "error": "WANDB_API_KEY not set"}
    body = {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You explain a bicycle-camera risk score using only the evidence. "
                    "Do not invent intent, meters, crashes, or probabilities. "
                    "Call the number a heuristic image-space score."
                ),
            },
            {"role": "user", "content": evidence},
        ],
        "max_tokens": 220,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {WANDB_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Project": f"{WANDB_TEAM}/{WANDB_PROJECT}".strip("/"),
    }
    errors = []
    for url in (
        "https://api.inference.wandb.ai/v1/chat/completions",
        "https://api.wandb.ai/proxy/weave/inference/v1/chat/completions",
    ):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=20)
            if response.status_code >= 400:
                errors.append(f"HTTP {response.status_code}")
                continue
            text = response.json()["choices"][0]["message"]["content"]
            return {"ok": True, "text": text}
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:120])
    return {"ok": False, "error": "; ".join(errors)[:240]}


def _public_meta(meta: dict) -> dict[str, Any]:
    return {
        "camera_id": meta.get("camera_id"),
        "location": meta.get("location"),
        "capture_type": meta.get("capture_type"),
        "duration": meta.get("duration"),
        "fps": meta.get("fps"),
        "segment_start_sec": meta.get("segment_start_sec"),
        "segment_end_sec": meta.get("segment_end_sec"),
        "segment_number": meta.get("segment_number"),
        "total_segments": meta.get("total_segments"),
        "filename": meta.get("filename"),
        "original_video": meta.get("original_video"),
        "object_classes": meta.get("object_classes"),
        "object_counts": meta.get("object_counts"),
        "reasoning_content": meta.get("reasoning_content") or "",
        "detection_count": meta.get("detection_count"),
        "detection_frame_count": meta.get("detection_frame_count"),
    }


CLIP_CACHE_MAX = 48


def load_clip(source: str, force: bool = False) -> dict[str, Any]:
    with _clips_lock:
        if not force and source in _clips:
            clip = _clips.pop(source)
            _clips[source] = clip
            return clip
    meta_r = _vss("GET", "/api/v1/videos/metadata", params={"source": source})
    if meta_r.status_code == 404:
        raise LookupError("That segment is not in the index.")
    meta_r.raise_for_status()
    meta = meta_r.json()
    if meta.get("camera_id") != BIKE_CAMERA or meta.get("location") != "new_york":
        raise ValueError("This clip is not NYC bike GoPro footage. GHOST NYC will not substitute other video.")
    det_r = _vss("GET", "/api/v1/videos/detections", params={"source": source})
    if det_r.status_code == 404:
        raise ValueError("No YOLO sidecar for this segment.")
    det_r.raise_for_status()
    analysis = analyze_detections(det_r.json())
    clip = {
        "source": source,
        "metadata": _public_meta(meta),
        "analysis": analysis,
        "loaded_at": time.time(),
    }
    with _clips_lock:
        _clips[source] = clip
        while len(_clips) > CLIP_CACHE_MAX:
            _clips.pop(next(iter(_clips)))
    _remember_summary(clip)
    return clip


def _events_from_timeline(timeline: list[dict]) -> list[dict]:
    """Local peaks of the max-risk curve at or above 50. Real timeline samples only."""
    events = []
    for i, point in enumerate(timeline):
        val = point.get("max_risk")
        if val is None or val < 50:
            continue
        prev = timeline[i - 1].get("max_risk") if i > 0 else None
        nxt = timeline[i + 1].get("max_risk") if i + 1 < len(timeline) else None
        if (prev is None or val >= prev) and (nxt is None or val > nxt):
            if events and point["t"] - events[-1]["t"] < 0.6:
                if val > events[-1]["max_risk"]:
                    events[-1] = {"t": point["t"], "max_risk": val, "conflict_count": point.get("conflict_count", 0)}
                continue
            events.append({"t": point["t"], "max_risk": val, "conflict_count": point.get("conflict_count", 0)})
    return events


def _summarize(clip: dict) -> dict[str, Any]:
    analysis = clip["analysis"]
    width, height = analysis["width"], analysis["height"]
    frames = analysis["frames"]
    occluded_t = None
    for fr in frames[::5]:
        if occlusion_zones(fr, width, height)["supported"]:
            occluded_t = fr["t"]
            break
    events = _events_from_timeline(analysis.get("timeline") or [])
    highlight = analysis.get("highlight") or {}
    vehicles = [
        row for row in analysis["tracks"] if row["label"] in ("car", "bus", "truck", "motorcycle") and row["max_risk"] is not None
    ]
    top_vehicle = max(vehicles, key=lambda r: r["max_risk"]) if vehicles else None
    return {
        "tracked_objects": analysis["tracked_objects"],
        "detection_count": analysis["detection_count"],
        "peak_risk": highlight.get("score"),
        "peak_t": highlight.get("t"),
        "peak_id": highlight.get("id"),
        "peak_label": highlight.get("label"),
        "events": len(events),
        "event_list": events[:12],
        "occlusion_t": occluded_t,
        "classes_seen": analysis["classes_seen"],
        "corridor_mode": analysis["corridor_mode"],
        "top_vehicle": {"id": top_vehicle["id"], "label": top_vehicle["display"], "risk": top_vehicle["max_risk"]} if top_vehicle else None,
    }


_summaries: dict[str, dict] = {}
_library_lock = threading.Lock()
_library: dict[str, Any] = {"status": "idle", "chunks_total": 0, "chunks_done": 0, "segments": {}, "summaries_done": 0}


def _remember_summary(clip: dict) -> None:
    try:
        summary = _summarize(clip)
    except Exception:  # noqa: BLE001
        return
    with _library_lock:
        _summaries[clip["source"]] = summary
        _library["summaries_done"] = len(_summaries)


def _parse_name(filename: str) -> dict[str, Any]:
    ride = "GOPR0130" if "GOPR0130" in filename else "GX010001" if "GX010001" in filename else "GX050001" if "GX050001" in filename else "ride"
    chunk_index = 0
    seg_no = 1
    try:
        chunk_index = int(filename.split("_chunk_")[1][:4])
    except (IndexError, ValueError):
        pass
    try:
        seg_no = int(filename.split("_segment_")[1][:3])
    except (IndexError, ValueError):
        pass
    return {"ride": ride, "chunk_index": chunk_index, "segment_number": seg_no}


def _categories(classes: str | None) -> list[str]:
    labels = {c.strip() for c in (classes or "").split(",") if c.strip()}
    cats = []
    if "person" in labels:
        cats.append("pedestrian")
    if labels & {"car", "bus", "truck", "motorcycle"}:
        cats.append("vehicle")
    if "bicycle" in labels:
        cats.append("cycling")
    return cats


def _build_library() -> None:
    try:
        catalog = _catalog()
    except Exception as exc:  # noqa: BLE001
        with _library_lock:
            _library["status"] = f"error: {exc}"
        return
    chunks = sorted((c for rows in catalog["rides"].values() for c in rows), key=lambda c: c.get("filename") or "")
    with _library_lock:
        _library["status"] = "loading"
        _library["chunks_total"] = len(chunks)
    for chunk in chunks:
        try:
            response = _vss("GET", "/api/v1/tools/segments", params={"original_video": chunk["original_video"]}, timeout=60)
            response.raise_for_status()
            for seg in response.json().get("segments") or []:
                if seg.get("camera_id") != BIKE_CAMERA:
                    continue
                name = seg.get("filename") or ""
                parsed = _parse_name(name)
                start = float(seg.get("segment_start_sec") or 0.0)
                ride_offset = parsed["chunk_index"] * 30.0 + start
                row = {
                    "source": seg.get("source"),
                    "filename": name,
                    "ride": parsed["ride"],
                    "chunk_index": parsed["chunk_index"],
                    "segment_number": parsed["segment_number"],
                    "total_segments": seg.get("total_segments"),
                    "ride_offset_sec": round(ride_offset, 1),
                    "title": f"{parsed['ride']} · {int(ride_offset // 60):02d}:{int(ride_offset % 60):02d}",
                    "duration": seg.get("duration"),
                    "camera_id": seg.get("camera_id"),
                    "location": seg.get("location"),
                    "capture_type": seg.get("capture_type"),
                    "object_classes": [c.strip() for c in (seg.get("object_classes") or "").split(",") if c.strip()],
                    "categories": _categories(seg.get("object_classes")),
                    "detection_count": seg.get("detection_count"),
                    "caption": (seg.get("reasoning_content") or "")[:300],
                    "upload_timestamp": seg.get("upload_timestamp"),
                    "original_video": seg.get("original_video"),
                }
                if row["source"]:
                    with _library_lock:
                        _library["segments"][row["source"]] = row
        except Exception as exc:  # noqa: BLE001
            print("library chunk failed:", exc, flush=True)
        with _library_lock:
            _library["chunks_done"] += 1
    with _library_lock:
        _library["status"] = "ready"
        order = sorted(_library["segments"], key=lambda s: (_library["segments"][s]["ride"], _library["segments"][s]["ride_offset_sec"]))
    # Summaries: default clip first, then the rest.
    for src in [DEFAULT_SOURCE] + [s for s in order if s != DEFAULT_SOURCE]:
        if src in _summaries:
            continue
        try:
            load_clip(src)
        except Exception as exc:  # noqa: BLE001
            print("summary failed:", src.rsplit("/", 1)[-1], exc, flush=True)
    print("library summaries complete", len(_summaries), flush=True)


def _library_rows() -> list[dict]:
    with _library_lock:
        rows = [dict(row) for row in _library["segments"].values()]
        for row in rows:
            summary = _summaries.get(row["source"])
            # Keep the library payload light: event_list is only needed per clip.
            row["summary"] = {k: v for k, v in summary.items() if k != "event_list"} if summary else None
            row["caption"] = (row.get("caption") or "")[:220]
            row["video_url"] = "api/video?source=" + quote(row["source"], safe="")
    rows.sort(key=lambda r: (r["ride"], r["ride_offset_sec"]))
    return rows


def _client_clip(clip: dict) -> dict[str, Any]:
    analysis = clip["analysis"]
    return {
        "source": clip["source"],
        "metadata": clip["metadata"],
        "video_url": "api/video?source=" + quote(clip["source"], safe=""),
        "mode": "offline_replay",
        "mode_label": "Offline analysis, replayed in sync with the clip",
        "fps": analysis["fps"],
        "width": analysis["width"],
        "height": analysis["height"],
        "frame_count": analysis["frame_count"],
        "detection_count": analysis["detection_count"],
        "tracked_objects": analysis["tracked_objects"],
        "classes_seen": analysis["classes_seen"],
        "formula": analysis["formula"],
        "weights": analysis["weights"],
        "assumptions": analysis["assumptions"],
        "corridor_mode": analysis["corridor_mode"],
        "highlight": analysis["highlight"],
        "timeline": analysis["timeline"],
        "tracks": analysis["tracks"],
        "frames": analysis["frames"],
        "score_note": "Percentages are normalized heuristic interaction scores, not crash probabilities.",
    }


def _catalog() -> dict[str, Any]:
    global _catalog_cache
    with _catalog_lock:
        if _catalog_cache and time.time() - _catalog_cache["fetched_at"] < 300:
            return _catalog_cache
    chunks = []
    offset = 0
    total = None
    while total is None or offset < total:
        response = _vss(
            "GET",
            "/api/v1/videos/explore",
            params={"scope": "all", "limit": 100, "offset": offset, "location": "new_york"},
        )
        response.raise_for_status()
        payload = response.json()
        total = int(payload.get("total") or 0)
        page = payload.get("chunks") or []
        if not page:
            break
        for chunk in page:
            if chunk.get("camera_id") != BIKE_CAMERA:
                continue
            chunks.append(
                {
                    "filename": chunk.get("filename"),
                    "original_video": chunk.get("original_video"),
                    "preview_source": chunk.get("preview_source"),
                    "chunk_duration_sec": chunk.get("chunk_duration_sec"),
                    "total_segments": chunk.get("total_segments"),
                    "camera_id": chunk.get("camera_id"),
                    "location": chunk.get("location"),
                    "capture_type": chunk.get("capture_type"),
                    "caption": (chunk.get("reasoning_content") or "")[:280],
                }
            )
        offset += len(page)
    chunks.sort(key=lambda item: item.get("filename") or "")
    rides: dict[str, list[dict]] = {}
    for chunk in chunks:
        name = chunk.get("filename") or ""
        ride = "GOPR0130" if "GOPR0130" in name else "GX010001" if "GX010001" in name else "GX050001" if "GX050001" in name else "other"
        rides.setdefault(ride, []).append(chunk)
    built = {
        "camera_id": BIKE_CAMERA,
        "location": "new_york",
        "chunk_count": len(chunks),
        "rides": rides,
        "note": (
            "Indexed as nyc_bike_gopro-1 in New York. "
            "Cosmos captions often describe a forward view along a bike lane with a cyclist ahead. "
            "This catalog does not include highway or other cities."
        ),
        "fetched_at": time.time(),
    }
    with _catalog_lock:
        _catalog_cache = built
    return built


def _object_at(clip: dict, track_id: int, t: float | None) -> tuple[dict | None, dict | None]:
    frames = clip["analysis"]["frames"]
    if not frames:
        return None, None
    if t is None:
        t = frames[0]["t"]
    best = min(frames, key=lambda fr: abs(fr["t"] - t))
    obj = next((item for item in best["objects"] if item["id"] == track_id), None)
    return best, obj


@app.get("/health")
def health():
    return jsonify({"status": "ok" if VSS_URL and VSS_USERNAME else "misconfigured", "app": "ghost-nyc"})


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/api/library")
def api_library():
    with _library_lock:
        status = _library["status"]
        total = _library["chunks_total"]
        done = _library["chunks_done"]
        summaries = _library["summaries_done"]
    return jsonify(
        {
            "status": status,
            "chunks_total": total,
            "chunks_done": done,
            "summaries_done": summaries,
            "camera_id": BIKE_CAMERA,
            "location_label": "new_york",
            "location_note": "Location is the ingest label on the camera, not a verified street address.",
            "segments": _library_rows(),
        }
    )


@app.get("/api/catalog")
def api_catalog():
    try:
        data = _catalog()
        return jsonify({k: v for k, v in data.items() if k != "fetched_at"})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/api/segments")
def api_segments():
    original = request.args.get("original_video") or ""
    if not original:
        return jsonify({"error": "original_video required"}), 400
    try:
        response = _vss("GET", "/api/v1/tools/segments", params={"original_video": original})
        response.raise_for_status()
        payload = response.json()
        rows = []
        for seg in payload.get("segments") or []:
            if seg.get("camera_id") != BIKE_CAMERA:
                continue
            rows.append(
                {
                    "filename": seg.get("filename"),
                    "source": seg.get("source"),
                    "duration": seg.get("duration"),
                    "segment_start_sec": seg.get("segment_start_sec"),
                    "segment_end_sec": seg.get("segment_end_sec"),
                    "segment_number": seg.get("segment_number"),
                    "object_classes": seg.get("object_classes"),
                    "object_counts": seg.get("object_counts"),
                    "detection_count": seg.get("detection_count"),
                    "caption": (seg.get("reasoning_content") or "")[:320],
                }
            )
        rows.sort(key=lambda item: item.get("segment_number") or 0)
        return jsonify({"original_video": original, "segments": rows})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/api/clip")
def api_clip():
    source = request.args.get("source") or DEFAULT_SOURCE
    try:
        clip = load_clip(source, force=request.args.get("refresh") == "1")
        return jsonify(_client_clip(clip))
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/explain")
def api_explain():
    body = request.get_json(force=True, silent=True) or {}
    source = body.get("source") or DEFAULT_SOURCE
    try:
        track_id = int(body.get("track_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "track_id required"}), 400
    try:
        t = float(body["t"]) if body.get("t") is not None else None
    except (TypeError, ValueError):
        return jsonify({"error": "t must be a number"}), 400
    try:
        clip = load_clip(source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    frame, obj = _object_at(clip, track_id, t)
    if obj is None:
        return jsonify({"error": "Object is not visible at that time."}), 404
    caption = clip["metadata"].get("reasoning_content") or ""
    text = explain_factors(obj, frame.get("corridor_mode") or clip["analysis"]["corridor_mode"], caption)
    track = next((row for row in clip["analysis"]["tracks"] if row["id"] == track_id), None)
    evidence = text
    if track and track.get("series"):
        tail = track["series"][-4:]
        evidence += " Recent scores: " + ", ".join(f"t={p['t']}s {p['risk']}%" for p in tail)
    if caption:
        evidence += " Caption excerpt: " + caption[:360]
    model = _wandb(evidence)
    return jsonify(
        {
            "text": text,
            "source": "wandb" if model.get("ok") else "deterministic",
            "model_note": model.get("text") if model.get("ok") else None,
            "model_error": None if model.get("ok") else model.get("error"),
            "object": obj,
            "track": track,
        }
    )


def _frame_obj(body: dict):
    source = body.get("source") or DEFAULT_SOURCE
    try:
        track_id = int(body.get("track_id"))
    except (TypeError, ValueError):
        raise ValueError("track_id required")
    try:
        t = float(body["t"]) if body.get("t") is not None else None
    except (TypeError, ValueError):
        raise ValueError("t must be a number")
    clip = load_clip(source)
    frame, obj = _object_at(clip, track_id, t)
    if frame is None or obj is None:
        raise LookupError("Object is not visible at that time.")
    return clip, frame, obj


def _bike_search(query: str) -> tuple[list[dict], str | None]:
    body = {
        "query": query,
        "top_k": 6,
        "llm_top_n": 1,
        "min_similarity": 0.18,
        "time_filter": "all",
        "metadata_filters": {"camera_id": BIKE_CAMERA, "location": "new_york"},
        "include_public": True,
    }
    response = _vss("POST", "/api/v1/search", json_body=body, timeout=60)
    if not response.ok:
        return [], f"Search returned HTTP {response.status_code}."
    hits = []
    for hit in response.json().get("results") or []:
        src = hit.get("source")
        if not src:
            continue
        hits.append(
            {
                "source": src,
                "filename": hit.get("filename") or src.rsplit("/", 1)[-1],
                "score": hit.get("similarity_score"),
                "camera_id": hit.get("camera_id"),
                "location": hit.get("location"),
                "t_start": hit.get("segment_start_sec"),
                "t_end": hit.get("segment_end_sec"),
                "caption": (hit.get("reasoning_content") or "")[:280],
                "objects": hit.get("object_classes"),
            }
        )
    return hits[:6], None


def _local_events(clip: dict) -> list[dict]:
    rows = []
    for point in clip["analysis"].get("timeline") or []:
        if (point.get("max_risk") or 0) < 45:
            continue
        rows.append(
            {
                "t": point["t"],
                "max_risk": point["max_risk"],
                "conflict_count": point["conflict_count"],
                "source": clip["source"],
                "filename": clip["metadata"].get("filename"),
                "camera_id": clip["metadata"].get("camera_id"),
                "location": clip["metadata"].get("location"),
            }
        )
    step = max(1, len(rows) // 8)
    return rows[::step][:8]


def _controls(body: dict) -> tuple[float, float, float]:
    try:
        delay = float(body.get("delay_s", 0.25))
        brake = float(body.get("brake", 0.4))
        horizon = float(body.get("horizon_s", 1.2))
    except (TypeError, ValueError):
        raise ValueError("delay_s, brake, and horizon_s must be numbers")
    return delay, brake, horizon


@app.post("/api/futures")
def api_futures():
    body = request.get_json(force=True, silent=True) or {}
    try:
        clip, frame, obj = _frame_obj(body)
        delay, brake, horizon = _controls(body)
        pack = simulate_futures(
            frame, obj, clip["analysis"]["width"], clip["analysis"]["height"], delay, brake, horizon
        )
        return jsonify(pack)
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/api/occlusion")
def api_occlusion():
    source = request.args.get("source") or DEFAULT_SOURCE
    try:
        t = float(request.args.get("t") or 0)
    except ValueError:
        return jsonify({"error": "t must be a number"}), 400
    try:
        clip = load_clip(source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    frames = clip["analysis"]["frames"]
    if not frames:
        return jsonify({"supported": False, "zones": [], "disclaimer": "No frames."})
    frame = min(frames, key=lambda fr: abs(fr["t"] - t))
    return jsonify(occlusion_zones(frame, clip["analysis"]["width"], clip["analysis"]["height"]))


@app.get("/api/occlusion/first")
def api_occlusion_first():
    source = request.args.get("source") or DEFAULT_SOURCE
    try:
        clip = load_clip(source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    width, height = clip["analysis"]["width"], clip["analysis"]["height"]
    for fr in clip["analysis"]["frames"][::3]:
        zones = occlusion_zones(fr, width, height)
        if zones["supported"]:
            return jsonify({"supported": True, "t": fr["t"], **zones})
    return jsonify({"supported": False, "t": None, "zones": [], "disclaimer": "No frame in this clip has a large, low vehicle box that hides the view beyond it."})


def _top_objects(frame: dict, limit: int = 3) -> list[dict]:
    rows = [o for o in frame.get("objects") or [] if o.get("risk") is not None]
    rows.sort(key=lambda o: -o["risk"])
    return rows[:limit]


@app.post("/api/ask")
def api_ask():
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400
    source = body.get("source") or DEFAULT_SOURCE
    try:
        clip = load_clip(source)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500
    frames = clip["analysis"]["frames"]
    try:
        t = float(body.get("t")) if body.get("t") is not None else (clip["analysis"].get("highlight") or {}).get("t") or 0.0
    except (TypeError, ValueError):
        t = 0.0
    frame = min(frames, key=lambda fr: abs(fr["t"] - t)) if frames else None
    obj = None
    if body.get("track_id") is not None and frame is not None:
        try:
            obj = next((o for o in frame["objects"] if o["id"] == int(body["track_id"])), None)
        except (TypeError, ValueError):
            obj = None
    if obj is None and frame is not None:
        top = _top_objects(frame, 1)
        obj = top[0] if top else None
    q = question.lower()
    meta = clip["metadata"]
    width, height = clip["analysis"]["width"], clip["analysis"]["height"]
    citations = [{"label": f"This clip at t={frame['t']}s" if frame else "This clip", "source": source, "t": frame["t"] if frame else 0}]
    parts = []
    technical: dict[str, Any] = {"frame_t": frame["t"] if frame else None, "corridor_mode": frame.get("corridor_mode") if frame else None}
    similar: list[dict] = []
    if obj is not None:
        technical["object"] = {k: obj.get(k) for k in ("id", "display", "risk", "status", "tracking", "conf", "hits", "factors")}
    if any(w in q for w in ("brake", "slow", "evade", "swerve", "future", "turn", "what if")):
        if obj is None:
            parts.append("No scored object is visible at this moment, so there is no future to project.")
        else:
            pack = simulate_futures(frame, obj, width, height)
            technical["futures"] = [{k: s[k] for k in ("id", "image_score", "level", "min_distance")} for s in pack["scenarios"]]
            by = {s["id"]: s for s in pack["scenarios"]}
            parts.append(
                f"For {obj['display']} #{obj['id']} at t={frame['t']}s, the image-space projections give: "
                + ", ".join(f"{s['title']} {s['image_score']} ({s['level']})" for s in pack["scenarios"])
                + f". Evading moves the path {pack['evade_direction']}. "
                + ("Braking lowers the image score here." if by["brake"]["image_score"] < by["continue"]["image_score"] else "Braking does not lower the image score here, because the shortened path stays near the object.")
                + " These are not distances in meters and not a certain collision."
            )
    if any(w in q for w in ("why", "risky", "risk", "danger")):
        if obj is None:
            parts.append("No object has a score at this moment. Objects need at least three tracked frames before GHOST scores them.")
        else:
            parts.append(explain_factors(obj, frame.get("corridor_mode") or "approximate", meta.get("reasoning_content") or ""))
    if any(w in q for w in ("what happened", "happen", "scene", "describe", "see")) and not parts:
        tops = _top_objects(frame) if frame else []
        parts.append(
            f"At t={frame['t'] if frame else 0}s the tracker holds {frame['tracked'] if frame else 0} road users. "
            + (
                "Highest scores: " + ", ".join(f"{o['display']} #{o['id']} {round(o['risk'])}" for o in tops) + "."
                if tops
                else "No object is scored at this moment."
            )
        )
        caption = meta.get("reasoning_content") or ""
        if caption:
            parts.append("Cosmos caption for this segment: " + caption[:360])
    if any(w in q for w in ("similar", "find", "before", "other", "search")):
        query = question
        if obj is not None and len(question.split()) < 4:
            query = f"{obj['display'].lower()} near a cyclist in a bike lane"
        similar, err = _bike_search(query)
        technical["search_query"] = query
        if similar:
            parts.append(
                f"Semantic search on the NYC bike camera returned {len(similar)} similar moments. They share the camera and the city label, not a verified street corner."
            )
            for hit in similar[:4]:
                citations.append({"label": f"{hit['filename']} t={hit.get('t_start')}s", "source": hit["source"], "t": 0, "caption": hit.get("caption")})
        else:
            parts.append("Semantic search returned no similar bike-camera moments." + (f" ({err})" if err else ""))
    if any(w in q for w in ("evidence", "support", "prove", "sure", "confident")):
        if obj is not None:
            f = obj.get("factors") or {}
            parts.append(
                f"Evidence for {obj['display']} #{obj['id']}: {obj.get('hits')} YOLO hits, confidence {obj.get('conf')}, tracking {obj.get('tracking')}. "
                f"Factor terms: overlap {f.get('trajectory_overlap')}, proximity {f.get('apparent_proximity')}, closing {f.get('closing_motion')}, timing {f.get('conflict_timing')}, certainty {f.get('certainty')}. "
                "The corridor is an estimate from image motion. Nothing here is a measured distance."
            )
        else:
            parts.append("The evidence available is the YOLO sidecar, the tracker output, and the Cosmos caption. No object is scored at this moment.")
    if not parts:
        tops = _top_objects(frame) if frame else []
        parts.append(
            "GHOST can explain object risk, project brake or evade futures, find similar moments, or list the evidence. "
            + ("Right now the highest scores are " + ", ".join(f"{o['display']} #{o['id']} {round(o['risk'])}" for o in tops) + "." if tops else "No object is scored at this moment.")
        )
    answer = " ".join(parts)
    evidence = (
        "QUESTION: " + question + "\nGROUNDED ANSWER: " + answer
        + "\nRULES: restate only this evidence in under 120 words. Do not add objects, meters, intent, or a crash claim. Call the number a heuristic risk score."
    )
    model = _wandb(evidence)
    return jsonify(
        {
            "question": question,
            "answer": answer,
            "model_note": model.get("text") if model.get("ok") else None,
            "source": "wandb" if model.get("ok") else "deterministic",
            "model_error": None if model.get("ok") else model.get("error"),
            "citations": citations,
            "technical": {**technical, "model_input": evidence},
            "similar": similar,
        }
    )


@app.post("/api/memory")
def api_memory():
    body = request.get_json(force=True, silent=True) or {}
    query = (body.get("query") or "bicycle conflict with a vehicle or pedestrian").strip()
    source = body.get("source") or DEFAULT_SOURCE
    note = (
        "Results share indexed camera nyc_bike_gopro-1 and location label new_york. "
        "That is not a verified street address, so these are similar moments, not the same corner."
    )
    if "taxi" in query.lower():
        note += " YOLO has no taxi class. This search uses captions, not a taxi detector."
    try:
        hits, err = _bike_search(query)
        local = []
        try:
            local = _local_events(load_clip(source))
        except Exception:
            local = []
        return jsonify({"query": query, "note": note, "search_error": err, "results": hits, "this_clip": local})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.post("/api/redesign")
def api_redesign():
    body = request.get_json(force=True, silent=True) or {}
    try:
        clip, frame, obj = _frame_obj(body)
        zones = occlusion_zones(frame, clip["analysis"]["width"], clip["analysis"]["height"])
        return jsonify(redesign_notes(frame, obj, zones))
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


def _case(body: dict) -> dict[str, Any]:
    clip, frame, obj = _frame_obj(body)
    delay, brake, horizon = _controls(body)
    width, height = clip["analysis"]["width"], clip["analysis"]["height"]
    futures = simulate_futures(frame, obj, width, height, delay, brake, horizon)
    zones = occlusion_zones(frame, width, height)
    redesign = redesign_notes(frame, obj, zones)
    query = body.get("query") or f"{obj.get('display')} near a cyclist in a bike lane"
    similar, search_error = _bike_search(query)
    brief = investigation_brief(clip["metadata"], frame, obj, futures, zones, similar, redesign)
    model = _wandb(brief)
    return {
        "brief": brief,
        "source": "wandb" if model.get("ok") else "deterministic",
        "model_note": model.get("text") if model.get("ok") else None,
        "model_error": None if model.get("ok") else model.get("error"),
        "futures": futures,
        "occlusion": zones,
        "redesign": redesign,
        "similar": similar,
        "search_error": search_error,
        "object": obj,
        "t": frame.get("t"),
        "metadata": {
            "filename": clip["metadata"].get("filename"),
            "camera_id": clip["metadata"].get("camera_id"),
            "location": clip["metadata"].get("location"),
            "source": clip["source"],
        },
    }


@app.post("/api/investigate")
def api_investigate():
    body = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify(_case(body))
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.get("/api/report")
def api_report():
    body = {
        "source": request.args.get("source") or DEFAULT_SOURCE,
        "track_id": request.args.get("track_id"),
        "t": request.args.get("t"),
    }
    try:
        case = _case(body)
    except LookupError as exc:
        return f"<pre>{exc}</pre>", 404
    except ValueError as exc:
        return f"<pre>{exc}</pre>", 400
    except Exception as exc:  # noqa: BLE001
        return f"<pre>{exc}</pre>", 500
    meta = case["metadata"]
    obj = case["object"]
    similar = "".join(
        f"<li>{h.get('filename')} · t={h.get('t_start')}s · {(h.get('caption') or '')[:180]}</li>" for h in case["similar"]
    )
    ideas = "".join(f"<li><strong>{i['title']}</strong> — {i['why']}</li>" for i in case["redesign"]["ideas"])
    futures = "".join(
        f"<li>{s['title']}: image score {s['image_score']}, {s['level']}, min image distance {s['min_distance']}</li>"
        for s in case["futures"]["scenarios"]
    )
    note = (case.get("model_note") or "Deterministic evidence only.").replace("<", "&lt;")
    brief = case["brief"].replace("<", "&lt;")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>GHOST NYC investigation</title>
<style>
body {{ font-family: Georgia, serif; color:#161616; margin:36px auto; max-width:800px; line-height:1.45; }}
h1 {{ font-family: "Avenir Next", sans-serif; letter-spacing:-0.03em; }}
.kicker {{ letter-spacing:0.14em; text-transform:uppercase; font-size:12px; color:#0a6f86; }}
td {{ border-top:1px solid #ddd; padding:6px 8px; vertical-align:top; }}
table {{ width:100%; border-collapse:collapse; }}
.note {{ background:#f4f7f8; padding:12px 14px; white-space:pre-wrap; }}
button {{ margin-top:16px; }}
@media print {{ button {{ display:none; }} }}
</style></head><body>
<p class="kicker">GHOST NYC · investigation · not a crash finding</p>
<h1>{obj.get('display')} #{obj.get('id')}</h1>
<table>
<tr><td>Clip</td><td>{meta.get('filename')}</td></tr>
<tr><td>Camera</td><td>{meta.get('camera_id')} · location label {meta.get('location')}</td></tr>
<tr><td>Time</td><td>t = {case.get('t')} s</td></tr>
<tr><td>Heuristic score</td><td>{obj.get('risk')} · {obj.get('status')}</td></tr>
</table>
<h2>Parallel futures</h2><ul>{futures}</ul>
<p>{case['futures']['disclaimer']}</p>
<h2>Conceptual design notes</h2><ul>{ideas}</ul>
<p>{case['redesign']['label']}</p>
<h2>Similar indexed moments</h2><ul>{similar or '<li>None returned.</li>'}</ul>
<h2>Evidence</h2><div class="note">{brief}</div>
<h2>Language model</h2><div class="note">{note}</div>
<button onclick="print()">Export PDF</button>
</body></html>"""


@app.get("/api/video")
def api_video():
    source = request.args.get("source") or DEFAULT_SOURCE
    token = _login()
    upstream = f"{VSS_URL}/api/v1/videos/stream?source={quote(source, safe='')}&token={quote(token)}"
    headers = {}
    if request.headers.get("Range"):
        headers["Range"] = request.headers["Range"]
    response = requests.get(upstream, headers=headers, stream=True, timeout=120)
    excluded = {"content-encoding", "transfer-encoding", "connection"}
    out = {k: v for k, v in response.headers.items() if k.lower() not in excluded}
    out.setdefault("Content-Type", "video/mp4")
    out["Accept-Ranges"] = "bytes"

    def generate():
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if chunk:
                yield chunk

    return Response(stream_with_context(generate()), status=response.status_code, headers=out, direct_passthrough=True)


def _warm() -> None:
    try:
        load_clip(DEFAULT_SOURCE)
        print("ghost-nyc clip cached", flush=True)
    except Exception as exc:  # noqa: BLE001
        print("ghost-nyc warm failed:", exc, flush=True)
    _build_library()


if __name__ == "__main__":
    threading.Thread(target=_warm, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), threaded=True)
