"""EditSpec: contract between what-if simulation and short video export.

Phase 0–1: geometric erase + paste along ghost path.
GENERATED pixels are hypothetical — never observed footage.
"""

from __future__ import annotations

from typing import Any

DEFAULT_EXPORT_DURATION_SEC = 2.0
MAX_EXPORT_DURATION_SEC = 5.0
DEFAULT_EXPORT_FPS = 5.0


class EditSpecError(ValueError):
    pass


def _clamp_duration(sec: float) -> float:
    return max(0.5, min(MAX_EXPORT_DURATION_SEC, float(sec)))


def validate_edit_spec(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise EditSpecError("edit_spec must be an object")
    source = str(raw.get("source") or "").strip()
    if not source:
        raise EditSpecError("source is required")
    try:
        event_t = float(raw.get("event_t"))
    except (TypeError, ValueError) as e:
        raise EditSpecError("event_t must be numeric") from e
    duration = _clamp_duration(raw.get("duration_sec", DEFAULT_EXPORT_DURATION_SEC))
    fps = float(raw.get("fps") or DEFAULT_EXPORT_FPS)
    fps = max(4.0, min(30.0, fps))
    tracks = raw.get("tracks") or []
    if not isinstance(tracks, list) or not tracks:
        raise EditSpecError("tracks must be a non-empty list")
    clean_tracks = []
    for tr in tracks:
        if not isinstance(tr, dict):
            raise EditSpecError("each track must be an object")
        tid = int(tr["track_id"])
        ghost = tr.get("ghost_path") or []
        if not isinstance(ghost, list) or len(ghost) < 2:
            raise EditSpecError(f"track {tid}: ghost_path needs ≥2 points")
        appearance = tr.get("appearance") or {}
        bbox = appearance.get("bbox")
        if not bbox or len(bbox) != 4:
            raise EditSpecError(f"track {tid}: appearance.bbox [x1,y1,x2,y2] required")
        clean_tracks.append(
            {
                "track_id": tid,
                "label": str(tr.get("label") or "vehicle"),
                "appearance": {
                    "t": float(appearance.get("t", event_t)),
                    "frame": int(appearance.get("frame", 0)),
                    "bbox": [float(x) for x in bbox],
                    "cx": float(appearance.get("cx", (bbox[0] + bbox[2]) / 2)),
                    "cy": float(appearance.get("cy", (bbox[1] + bbox[3]) / 2)),
                },
                "observed_boxes": list(tr.get("observed_boxes") or []),
                "ghost_path": [
                    {
                        "t": float(p["t"]),
                        "frame": int(p.get("frame", 0)),
                        "cx": float(p["cx"]),
                        "cy": float(p["cy"]),
                    }
                    for p in ghost
                ],
            }
        )
    return {
        "version": 1,
        "source": source,
        "event_t": event_t,
        "duration_sec": duration,
        "fps": fps,
        "run_index": int(raw.get("run_index", 0)),
        "seed": int(raw.get("seed", 0)),
        "scenario": raw.get("scenario") or {},
        "tracks": clean_tracks,
        "mode": str(raw.get("mode") or "geometric_move"),
        "label": "HYPOTHETICAL",
        "disclaimer": (
            "GENERATED edit: real vehicle pixels cropped from the clip and pasted along a "
            "hypothetical ghost path on real road frames. The motion after the event is not "
            "observed. Not a crash prediction or finding of fault. Image-space only."
        ),
    }


def _appearance_at_event(track: dict, event_t: float, event_frame: int) -> dict[str, Any]:
    path = track.get("full_path") or track.get("path") or []
    if not path:
        raise EditSpecError(f"track {track.get('id')}: empty path")
    # Prefer last point at or before event
    prior = [p for p in path if float(p.get("t", 0)) <= event_t + 1e-6]
    p = prior[-1] if prior else path[0]
    bbox = list(p.get("bbox") or [p["cx"] - 20, p["cy"] - 10, p["cx"] + 20, p["cy"] + 10])
    return {
        "t": float(p["t"]),
        "frame": int(p.get("frame", event_frame)),
        "bbox": bbox,
        "cx": float(p["cx"]),
        "cy": float(p["cy"]),
    }


def _observed_boxes(track: dict, t0: float, t1: float) -> list[dict[str, Any]]:
    out = []
    for p in track.get("full_path") or track.get("path") or []:
        t = float(p.get("t", 0))
        if t0 - 1e-6 <= t <= t1 + 1e-6:
            bbox = p.get("bbox") or [p["cx"] - 20, p["cy"] - 10, p["cx"] + 20, p["cy"] + 10]
            out.append({"t": t, "frame": int(p.get("frame", 0)), "bbox": list(bbox)})
    return out


def build_edit_spec(
    *,
    source: str,
    analysis: dict[str, Any],
    event: dict[str, Any],
    scenario: dict[str, Any],
    ensemble_result: dict[str, Any],
    run_index: int | None = None,
    duration_sec: float = DEFAULT_EXPORT_DURATION_SEC,
    fps: float = DEFAULT_EXPORT_FPS,
) -> dict[str, Any]:
    """Build a validated EditSpec from analysis + one ensemble run's ghosts."""
    if not ensemble_result.get("ok"):
        raise EditSpecError("ensemble_result.ok must be true")
    runs = ensemble_result.get("runs") or []
    if not runs:
        raise EditSpecError("ensemble_result has no runs")
    if run_index is None:
        run_index = int((ensemble_result.get("summary") or {}).get("most_interacting_run") or 0)
    run_index = max(0, min(len(runs) - 1, int(run_index)))
    run = runs[run_index]
    ghosts = {int(g["track_id"]): g for g in (run.get("ghosts") or []) if g.get("ghost_path")}
    if not ghosts:
        raise EditSpecError("selected run has no ghost paths")

    event_t = float(event.get("t") or 0)
    event_frame = int(event.get("frame") or 0)
    tracks_by_id = {int(t["id"]): t for t in (analysis.get("tracks") or [])}

    # Prefer scenario targets; else all ghosts
    targets = list((scenario or {}).get("targetTrackIds") or [])
    if not targets:
        targets = list(ghosts.keys())

    edit_tracks = []
    for tid in targets:
        if tid not in ghosts:
            continue
        if tid not in tracks_by_id:
            raise EditSpecError(f"ghost track {tid} missing from analysis.tracks")
        tr = tracks_by_id[tid]
        appearance = _appearance_at_event(tr, event_t, event_frame)
        t_end = event_t + _clamp_duration(duration_sec)
        edit_tracks.append(
            {
                "track_id": tid,
                "label": tr.get("label") or ghosts[tid].get("label") or "vehicle",
                "appearance": appearance,
                "observed_boxes": _observed_boxes(tr, event_t, t_end),
                "ghost_path": ghosts[tid]["ghost_path"],
            }
        )
    if not edit_tracks:
        raise EditSpecError("no editable tracks resolved from targets ∩ ghosts")

    # Neighbor vehicles (not moved) — used to protect them from erase/inpaint smear
    target_set = {tr["track_id"] for tr in edit_tracks}
    neighbor_boxes: list[dict[str, Any]] = []
    for tid, tr in tracks_by_id.items():
        if tid in target_set:
            continue
        try:
            app = _appearance_at_event(tr, event_t, event_frame)
            neighbor_boxes.append(
                {
                    "track_id": tid,
                    "bbox": list(app["bbox"]),
                    "observed_boxes": _observed_boxes(tr, event_t, event_t + _clamp_duration(duration_sec)),
                }
            )
        except EditSpecError:
            continue

    spec = validate_edit_spec(
        {
            "source": source,
            "event_t": event_t,
            "duration_sec": duration_sec,
            "fps": fps,
            "run_index": run_index,
            "seed": int(run.get("seed") or ensemble_result.get("seed") or 0),
            "scenario": scenario,
            "tracks": edit_tracks,
            "mode": "geometric_move",
        }
    )
    spec["neighbor_boxes"] = neighbor_boxes
    return spec
