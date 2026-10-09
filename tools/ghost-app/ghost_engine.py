"""GHOST engine: per-video vehicle tracking + image-space counterfactuals.

All geometry is in pixel / frame coordinates unless noted.
This is NOT calibrated real-world physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

VEHICLE_LABELS = {"car", "truck", "bus", "motorcycle"}


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-9)


def center(bbox: list[float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def nms(dets: list[tuple], thr: float = 0.65) -> list[tuple]:
    dets = sorted(dets, key=lambda d: -d[2])
    keep: list[tuple] = []
    for d in dets:
        if all(iou(d[0], k[0]) < thr for k in keep):
            keep.append(d)
    return keep


@dataclass
class Track:
    id: int
    label: str
    bbox: list[float]
    conf: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def push(self, frame_i: int, t: float, bbox: list[float], conf: float) -> None:
        cx, cy = center(bbox)
        self.history.append(
            {
                "frame": frame_i,
                "t": t,
                "cx": cx,
                "cy": cy,
                "bbox": list(bbox),
                "conf": conf,
            }
        )
        self.bbox = list(bbox)
        self.conf = conf

    def update(self, bbox: list[float], conf: float, frame_i: int, t: float) -> None:
        self.push(frame_i, t, bbox, conf)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.age += 1


def track_vehicles(
    frames: list[dict],
    *,
    iou_thresh: float = 0.25,
    max_age: int = 20,
    min_conf: float = 0.4,
    min_hits: int = 8,
) -> list[Track]:
    """ByteTrack-lite: greedy IoU association on NMS'd vehicle detections.

    Temporary IDs are valid only within this video segment.
    """
    tracks: list[Track] = []
    finished: list[Track] = []
    next_id = 1

    for fr in frames:
        fi = int(fr.get("frame_index", 0))
        t = float(fr.get("time_sec", fi / 30.0))
        dets = [
            (list(d["bbox"]), d["label"], float(d["confidence"]))
            for d in fr.get("detections") or []
            if d.get("label") in VEHICLE_LABELS and float(d.get("confidence") or 0) >= min_conf
        ]
        dets = nms(dets, 0.65)

        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(tracks):
            for di, (bb, _lab, _cf) in enumerate(dets):
                pairs.append((iou(tr.bbox, bb), ti, di))
        pairs.sort(reverse=True)

        used_t: set[int] = set()
        used_d: set[int] = set()
        for sc, ti, di in pairs:
            if sc < iou_thresh:
                break
            if ti in used_t or di in used_d:
                continue
            bb, _lab, cf = dets[di]
            tracks[ti].update(bb, cf, fi, t)
            used_t.add(ti)
            used_d.add(di)

        for ti, tr in enumerate(tracks):
            if ti not in used_t:
                tr.mark_missed()

        for di, (bb, lab, cf) in enumerate(dets):
            if di in used_d:
                continue
            tr = Track(id=next_id, label=lab, bbox=list(bb), conf=cf)
            next_id += 1
            tr.push(fi, t, bb, cf)
            tracks.append(tr)

        still: list[Track] = []
        for tr in tracks:
            if tr.time_since_update > max_age:
                if tr.hits >= min_hits:
                    finished.append(tr)
            else:
                still.append(tr)
        tracks = still

    for tr in tracks:
        if tr.hits >= min_hits:
            finished.append(tr)
    return finished


def _velocity_at(hist: dict[int, dict], frame: int, lookback: int = 5) -> tuple[float, float] | None:
    if frame not in hist:
        return None
    p = hist[frame]
    back = None
    for df in range(lookback, 0, -1):
        if frame - df in hist:
            back = hist[frame - df]
            break
    if back is None:
        return None
    dt = p["t"] - back["t"]
    if dt <= 1e-6:
        return None
    return ((p["cx"] - back["cx"]) / dt, (p["cy"] - back["cy"]) / dt)


def detect_events(tracks: list[Track], fps: float = 30.0) -> list[dict]:
    """Find close interactions + braking candidates in image space."""
    hist = {tr.id: {p["frame"]: p for p in tr.history} for tr in tracks}
    labels = {tr.id: tr.label for tr in tracks}
    ids = [tr.id for tr in tracks]
    events: list[dict] = []

    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            common = sorted(set(hist[a]) & set(hist[b]))
            if len(common) < 5:
                continue
            best = None
            for fi in common:
                pa, pb = hist[a][fi], hist[b][fi]
                if iou(pa["bbox"], pb["bbox"]) > 0.15:
                    continue
                d = math.hypot(pa["cx"] - pb["cx"], pa["cy"] - pb["cy"])
                aw = pa["bbox"][2] - pa["bbox"][0]
                ah = pa["bbox"][3] - pa["bbox"][1]
                bw = pb["bbox"][2] - pb["bbox"][0]
                bh = pb["bbox"][3] - pb["bbox"][1]
                scale = 0.5 * (math.hypot(aw, ah) + math.hypot(bw, bh))
                nd = d / max(scale, 1.0)
                if best is None or nd < best[0]:
                    best = (nd, d, pa["t"], fi, pa, pb)
            if not best or best[0] >= 2.2:
                continue
            nd, d, t, fi, pa, pb = best
            va = _velocity_at(hist[a], fi)
            vb = _velocity_at(hist[b], fi)
            ttc = None
            closing = None
            if va and vb:
                dx, dy = pb["cx"] - pa["cx"], pb["cy"] - pa["cy"]
                dist = math.hypot(dx, dy) + 1e-6
                closing = -((va[0] - vb[0]) * dx + (va[1] - vb[1]) * dy) / dist
                if closing > 5:
                    ttc = dist / closing
            kind = "close_interaction"
            if ttc is not None and ttc < 1.5:
                kind = "near_miss_candidate"
            events.append(
                {
                    "kind": kind,
                    "t": t,
                    "frame": fi,
                    "id_a": a,
                    "id_b": b,
                    "label_a": labels[a],
                    "label_b": labels[b],
                    "norm_dist": round(nd, 3),
                    "px_dist": round(d, 1),
                    "ttc_sec_image_space": None if ttc is None else round(ttc, 3),
                    "closing_px_per_sec": None if closing is None else round(closing, 1),
                    "units": "image-space pixels / seconds — not real-world meters",
                }
            )

    # Braking: speed drop within a track
    for tr in tracks:
        h = tr.history
        if len(h) < 12:
            continue
        speeds = []
        for i in range(1, len(h)):
            dt = h[i]["t"] - h[i - 1]["t"] or (1.0 / fps)
            sp = math.hypot(h[i]["cx"] - h[i - 1]["cx"], h[i]["cy"] - h[i - 1]["cy"]) / dt
            speeds.append((h[i]["t"], h[i]["frame"], sp))
        n = len(speeds)
        early = sum(s[2] for s in speeds[: n // 3]) / max(1, n // 3)
        late = sum(s[2] for s in speeds[-n // 3 :]) / max(1, n // 3)
        if early > 40 and late < early * 0.55:
            mid = speeds[n // 2]
            events.append(
                {
                    "kind": "braking_candidate",
                    "t": mid[0],
                    "frame": mid[1],
                    "id_a": tr.id,
                    "id_b": None,
                    "label_a": tr.label,
                    "label_b": None,
                    "speed_early_px_s": round(early, 1),
                    "speed_late_px_s": round(late, 1),
                    "units": "image-space pixels / seconds — not real-world meters",
                }
            )

    def sort_key(e: dict) -> tuple:
        ttc = e.get("ttc_sec_image_space")
        return (
            0 if e["kind"] == "near_miss_candidate" else 1 if e["kind"] == "close_interaction" else 2,
            ttc if ttc is not None else 99.0,
            e.get("norm_dist") or 99.0,
        )

    events.sort(key=sort_key)
    # Dedup similar pairs
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for e in events:
        key = (e["kind"], e.get("id_a"), e.get("id_b"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq[:20]


def ghost_trajectory(
    track: Track,
    event_frame: int,
    *,
    horizon_sec: float = 1.5,
    fps: float = 30.0,
) -> dict | None:
    """Project constant-velocity path from pre-event motion (no-brake / no-swerve ghost)."""
    hist = {p["frame"]: p for p in track.history}
    if event_frame not in hist:
        # nearest prior
        priors = [f for f in hist if f <= event_frame]
        if not priors:
            return None
        event_frame = max(priors)
    p0 = hist[event_frame]
    vel = _velocity_at(hist, event_frame, lookback=8)
    if not vel:
        return None
    vx, vy = vel
    steps = max(3, int(horizon_sec * fps))
    ghost = []
    for i in range(steps + 1):
        dt = i / fps
        ghost.append(
            {
                "t": p0["t"] + dt,
                "frame": event_frame + i,
                "cx": p0["cx"] + vx * dt,
                "cy": p0["cy"] + vy * dt,
            }
        )
    # Actual continuation after event for comparison
    actual = [p for p in track.history if p["frame"] >= event_frame]
    return {
        "track_id": track.id,
        "label": track.label,
        "event_frame": event_frame,
        "event_t": p0["t"],
        "velocity_px_s": [round(vx, 1), round(vy, 1)],
        "ghost_path": ghost,
        "actual_path": [
            {"t": p["t"], "frame": p["frame"], "cx": p["cx"], "cy": p["cy"]} for p in actual[: steps + 1]
        ],
        "assumption": "constant image-space velocity (no brake / no swerve)",
        "units": "pixels and seconds in camera image space",
    }


def assess_path_interaction(
    path_a: list[dict],
    path_b: list[dict],
    *,
    converge_norm: float = 1.15,
) -> dict:
    """Image-space convergence / separation assessment between two ghost paths.

    Returns kind in {converge, separated, uncertain}. Not a crash prediction.
    """
    if not path_a or not path_b:
        return {
            "outcome": "uncertain",
            "min_norm_dist": None,
            "note": "Insufficient path points for assessment",
        }
    # Align by index (same horizon sampling)
    n = min(len(path_a), len(path_b))
    best_nd = None
    best_t = None
    for i in range(n):
        pa, pb = path_a[i], path_b[i]
        d = math.hypot(pa["cx"] - pb["cx"], pa["cy"] - pb["cy"])
        # approximate scale from initial separation if bbox absent
        scale = max(d / 2.0, 40.0) if i == 0 else max(best_nd * 40.0 if best_nd else 40.0, 40.0)
        # Prefer bbox-derived scale when present on first points
        aw = pa.get("bbox")
        bw = pb.get("bbox")
        if aw and bw:
            scale = 0.5 * (
                math.hypot(aw[2] - aw[0], aw[3] - aw[1])
                + math.hypot(bw[2] - bw[0], bw[3] - bw[1])
            )
        nd = d / max(scale, 1.0)
        if best_nd is None or nd < best_nd:
            best_nd = nd
            best_t = pa.get("t")
    if best_nd is None:
        outcome = "uncertain"
    elif best_nd < converge_norm:
        outcome = "converge"
    elif best_nd > converge_norm * 1.6:
        outcome = "separated"
    else:
        outcome = "uncertain"
    return {
        "outcome": outcome,
        "min_norm_dist": None if best_nd is None else round(best_nd, 3),
        "t_at_min": best_t,
        "units": "image-space normalized separation — not a crash finding",
        "label": "HYPOTHETICAL",
    }


def decision_to_trajectory(
    track: Track,
    event_frame: int,
    decision: dict,
    *,
    horizon_sec: float = 1.5,
    fps: float = 30.0,
) -> dict | None:
    """Map a discrete driver decision into a deterministic image-space ghost path.

    Agents never output coordinates; this function owns kinematics.
    """
    from persona_config import (
        MAX_ACCEL_FRAC,
        MAX_DECEL_FRAC,
        MAX_LATERAL_RATE_FRAC,
        MAX_REACTION_DELAY_SEC,
        MIN_REACTION_DELAY_SEC,
    )

    hist = {p["frame"]: p for p in track.history}
    if event_frame not in hist:
        priors = [f for f in hist if f <= event_frame]
        if not priors:
            return None
        event_frame = max(priors)
    p0 = hist[event_frame]
    vel = _velocity_at(hist, event_frame, lookback=8)
    if not vel:
        return None
    vx, vy = vel
    speed = math.hypot(vx, vy) + 1e-6

    action = decision.get("action", "maintain")
    intensity = max(0.0, min(1.0, float(decision.get("intensity", 0.5))))
    delay = float(decision.get("reactionDelaySec", decision.get("reaction_delay_sec", 0.3)))
    delay = max(MIN_REACTION_DELAY_SEC, min(MAX_REACTION_DELAY_SEC, delay))

    # Baseline: maintain observed velocity until delay, then apply action.
    steps = max(3, int(horizon_sec * fps))
    ghost = []
    cx, cy = p0["cx"], p0["cy"]
    cur_vx, cur_vy = vx, vy
    applied = False
    for i in range(steps + 1):
        t_rel = i / fps
        if (not applied) and t_rel >= delay:
            applied = True
            if action == "brake":
                frac = min(MAX_DECEL_FRAC, intensity)
                cur_vx *= 1.0 - frac
                cur_vy *= 1.0 - frac
            elif action == "accelerate":
                frac = min(MAX_ACCEL_FRAC, intensity * MAX_ACCEL_FRAC)
                cur_vx *= 1.0 + frac
                cur_vy *= 1.0 + frac
            elif action == "steer_left":
                # image-space: left ≈ negative x (camera-dependent; labeled hypothetical)
                lat = min(MAX_LATERAL_RATE_FRAC, intensity * MAX_LATERAL_RATE_FRAC) * speed
                cur_vx -= lat
            elif action == "steer_right":
                lat = min(MAX_LATERAL_RATE_FRAC, intensity * MAX_LATERAL_RATE_FRAC) * speed
                cur_vx += lat
            elif action == "abort_lane_change":
                # pull heading back toward pre-event velocity
                cur_vx = vx * (1.0 - 0.5 * intensity) + cur_vx * (0.5 * intensity)
                cur_vy = vy
            # maintain: no change
        if i > 0:
            cx += cur_vx / fps
            cy += cur_vy / fps
        ghost.append(
            {
                "t": p0["t"] + t_rel,
                "frame": event_frame + i,
                "cx": cx,
                "cy": cy,
            }
        )

    actual = [p for p in track.history if p["frame"] >= event_frame]
    return {
        "track_id": track.id,
        "label": track.label,
        "event_frame": event_frame,
        "event_t": p0["t"],
        "ghost_path": ghost,
        "actual_path": [
            {"t": p["t"], "frame": p["frame"], "cx": p["cx"], "cy": p["cy"]} for p in actual[: steps + 1]
        ],
        "assumption": f"HYPOTHETICAL decision→trajectory: action={action}, intensity={intensity}, delay={delay}s",
        "decision": {
            "action": action,
            "intensity": intensity,
            "reactionDelaySec": delay,
        },
        "units": "pixels and seconds in camera image space",
        "label": "HYPOTHETICAL",
    }


def intervention_to_trajectory(
    track: Track,
    event_frame: int,
    scenario: dict,
    *,
    fps: float = 30.0,
) -> dict | None:
    """Route a validated scenario intervention through deterministic kinematics."""
    intervention = scenario.get("intervention")
    params = scenario.get("params") or {}
    horizon = float(scenario.get("horizonSec") or 1.5)

    if intervention == "exclude_vehicle":
        return {
            "track_id": track.id,
            "label": track.label,
            "event_frame": event_frame,
            "event_t": None,
            "ghost_path": [],
            "actual_path": [],
            "assumption": "HYPOTHETICAL: vehicle excluded from interaction",
            "excluded": True,
            "label": "HYPOTHETICAL",
        }

    # Map intervention → discrete decision, then reuse decision_to_trajectory
    delay = float(params.get("delay_sec", params.get("delaySec", 0.3)))
    brake_i = float(params.get("brake_intensity", params.get("brakeIntensity", 0.7)))
    steer_i = float(params.get("steer_intensity", params.get("steerIntensity", 0.4)))

    decision: dict[str, Any]
    if intervention == "maintain_velocity":
        decision = {"action": "maintain", "intensity": 0.0, "reactionDelaySec": 0.0}
    elif intervention == "maintain_heading":
        decision = {"action": "maintain", "intensity": 0.0, "reactionDelaySec": 0.0}
    elif intervention == "delayed_reaction":
        decision = {"action": "brake", "intensity": brake_i, "reactionDelaySec": max(0.5, delay)}
    elif intervention == "earlier_braking":
        decision = {"action": "brake", "intensity": brake_i, "reactionDelaySec": min(0.15, delay)}
    elif intervention == "later_braking":
        decision = {"action": "brake", "intensity": brake_i, "reactionDelaySec": max(0.6, delay)}
    elif intervention == "stronger_braking":
        decision = {"action": "brake", "intensity": max(0.85, brake_i), "reactionDelaySec": delay}
    elif intervention == "weaker_braking":
        decision = {"action": "brake", "intensity": min(0.35, brake_i), "reactionDelaySec": delay}
    elif intervention == "delayed_lane_change":
        decision = {"action": "steer_left", "intensity": steer_i, "reactionDelaySec": max(0.5, delay)}
    elif intervention == "aborted_lane_change":
        decision = {"action": "abort_lane_change", "intensity": max(0.5, steer_i), "reactionDelaySec": delay}
    else:
        # fallback constant-velocity ghost
        return ghost_trajectory(track, event_frame, horizon_sec=horizon, fps=fps)

    out = decision_to_trajectory(track, event_frame, decision, horizon_sec=horizon, fps=fps)
    if out:
        out["assumption"] = (
            f"HYPOTHETICAL intervention={intervention}; "
            + out.get("assumption", "")
        )
        out["scenario_intervention"] = intervention
    return out


def nearby_track_ids(
    tracks: list[Track],
    event: dict,
    *,
    max_nearby: int = 4,
    radius_norm: float = 4.0,
) -> list[int]:
    """Tracks near the event pair at the event frame (interaction neighborhood)."""
    hist = {tr.id: {p["frame"]: p for p in tr.history} for tr in tracks}
    fi = int(event.get("frame") or 0)
    anchors = [event.get("id_a"), event.get("id_b")]
    anchors = [a for a in anchors if a is not None and a in hist and fi in hist[a]]
    if not anchors:
        return []
    anchor_pts = [hist[a][fi] for a in anchors]
    scored: list[tuple[float, int]] = []
    for tr in tracks:
        if tr.id in anchors:
            continue
        if fi not in hist[tr.id]:
            continue
        p = hist[tr.id][fi]
        d = min(math.hypot(p["cx"] - a["cx"], p["cy"] - a["cy"]) for a in anchor_pts)
        scale = math.hypot(p["bbox"][2] - p["bbox"][0], p["bbox"][3] - p["bbox"][1])
        nd = d / max(scale, 1.0)
        if nd <= radius_norm:
            scored.append((nd, tr.id))
    scored.sort()
    return [tid for _nd, tid in scored[:max_nearby]]


def analyze_detections(det_payload: dict) -> dict:
    frames = det_payload.get("frames") or []
    fps = float(det_payload.get("fps") or 30.0)
    shape = det_payload.get("video_shape") or [1080, 1920]
    tracks = track_vehicles(frames)
    events = detect_events(tracks, fps=fps)

    track_by_id = {tr.id: tr for tr in tracks}
    ghosts = []
    primary = events[0] if events else None
    if primary:
        for tid in (primary.get("id_a"), primary.get("id_b")):
            if tid is None or tid not in track_by_id:
                continue
            g = ghost_trajectory(track_by_id[tid], int(primary["frame"]), fps=fps)
            if g:
                ghosts.append(g)

    # Compact track payload for UI (downsample history)
    track_out = []
    for tr in tracks:
        h = tr.history
        step = max(1, len(h) // 40)
        sampled = h[::step]
        if h and sampled[-1] is not h[-1]:
            sampled.append(h[-1])
        track_out.append(
            {
                "id": tr.id,
                "label": tr.label,
                "hits": tr.hits,
                "path": [
                    {
                        "frame": p["frame"],
                        "t": p["t"],
                        "cx": round(p["cx"], 1),
                        "cy": round(p["cy"], 1),
                        "bbox": [round(x, 1) for x in p["bbox"]],
                    }
                    for p in sampled
                ],
                "full_path": [
                    {
                        "frame": p["frame"],
                        "t": p["t"],
                        "cx": round(p["cx"], 1),
                        "cy": round(p["cy"], 1),
                        "bbox": [round(x, 1) for x in p["bbox"]],
                    }
                    for p in h
                ],
            }
        )

    return {
        "fps": fps,
        "video_shape": shape,  # [H, W]
        "frame_count": det_payload.get("frame_count") or len(frames),
        "n_tracks": len(tracks),
        "tracks": track_out,
        "events": events,
        "primary_event": primary,
        "ghosts": ghosts,
        "caveats": [
            "Tracking IDs are temporary and valid only within this segment.",
            "No cross-camera identity linking is performed.",
            "TTC and velocities are image-space estimates (pixels), not meters or calibrated world speed.",
            "Ghost paths assume constant image velocity (no brake / no swerve).",
            "YOLO can mislabel (e.g. train/airplane on highway); vehicle filter mitigates but does not eliminate this.",
        ],
    }
