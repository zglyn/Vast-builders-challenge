"""Image-space futures, occlusion wedges, and conceptual redesign notes.

Nothing here is a meter, a crash, or a measured safety benefit.
The cyclist path is the estimated corridor. Braking and evading move that
path only. Other objects keep the velocity already estimated from the track.
"""

from __future__ import annotations

import math
from typing import Any

from risk_engine import clip as clamp

STEPS = 9
OCCLUDERS = {"car", "bus", "truck"}


def _centerline(corridor: list[list[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    base = ((corridor[0][0] + corridor[1][0]) / 2.0, (corridor[0][1] + corridor[1][1]) / 2.0)
    far = ((corridor[2][0] + corridor[3][0]) / 2.0, (corridor[2][1] + corridor[3][1]) / 2.0)
    return base, far


def _norm_center(obj: dict, width: float, height: float) -> tuple[float, float]:
    box = obj["bbox"]
    return ((box[0] + box[2]) / 2.0 / width, (box[1] + box[3]) / 2.0 / height)


def object_samples(obj: dict, width: float, height: float, horizon: float, steps: int) -> list[list[float]]:
    """Resample the track's own predicted points. Extend with the last step if needed."""
    traj = obj.get("trajectory") or []
    cx, cy = _norm_center(obj, width, height)
    if len(traj) < 2:
        return [[round(cx, 4), round(cy, 4)] for _ in range(steps)]
    dt = 0.12
    times = [i * dt for i in range(len(traj))]
    vx = (traj[-1][0] - traj[-2][0]) / dt
    vy = (traj[-1][1] - traj[-2][1]) / dt
    pts = []
    last = steps - 1
    for i in range(steps):
        t = 0.0 if last == 0 else horizon * i / last
        if t <= times[-1]:
            j = 0
            while j < len(times) - 2 and times[j + 1] < t:
                j += 1
            span = max(1e-6, times[j + 1] - times[j])
            a = (t - times[j]) / span
            x = traj[j][0] + a * (traj[j + 1][0] - traj[j][0])
            y = traj[j][1] + a * (traj[j + 1][1] - traj[j][1])
        else:
            extra = t - times[-1]
            x = traj[-1][0] + vx * extra
            y = traj[-1][1] + vy * extra
        pts.append([round(clamp(x, 0.0, 1.0), 4), round(clamp(y, 0.0, 1.0), 4)])
    return pts


def _cyclist_path(
    base: tuple[float, float],
    far: tuple[float, float],
    mode: str,
    delay_s: float,
    brake: float,
    horizon: float,
    shift_sign: float,
    steps: int,
) -> list[list[float]]:
    pts = []
    last = steps - 1
    for i in range(steps):
        t = 0.0 if last == 0 else horizon * i / last
        if t <= delay_s or mode == "continue":
            prog = t / horizon
            shift = 0.0
        else:
            u = (t - delay_s) / max(0.05, horizon - delay_s)
            if mode == "brake":
                prog = (delay_s + (t - delay_s) * brake) / horizon
                shift = 0.0
            else:
                prog = t / horizon
                shift = shift_sign * 0.18 * u
        x = base[0] + (far[0] - base[0]) * prog + shift
        y = base[1] + (far[1] - base[1]) * prog
        pts.append([round(clamp(x, 0.0, 1.0), 4), round(clamp(y, 0.0, 1.0), 4)])
    return pts


def _conflict(cyclist: list[list[float]], other: list[list[float]], uncertainty: float) -> dict[str, Any]:
    best = 9.0
    at = 0
    for i, (c, o) in enumerate(zip(cyclist, other)):
        dist = math.hypot(c[0] - o[0], c[1] - o[1])
        if dist < best:
            best = dist
            at = i
    band_radius = 0.035 + uncertainty
    if best <= band_radius:
        level = "overlap"
    elif best <= band_radius + 0.06:
        level = "near"
    else:
        level = "clear"
    # Display score only: closer image paths score higher. Not a crash probability.
    score = int(round(clamp(100.0 * (1.0 - best / 0.28), 0.0, 100.0)))
    return {
        "min_distance": round(best, 4),
        "at_index": at,
        "level": level,
        "image_score": score,
    }


def simulate_futures(
    frame: dict,
    obj: dict,
    width: float,
    height: float,
    delay_s: float = 0.25,
    brake: float = 0.4,
    horizon_s: float = 1.2,
) -> dict[str, Any]:
    delay_s = clamp(float(delay_s), 0.0, 1.5)
    brake = clamp(float(brake), 0.05, 1.0)
    horizon_s = clamp(float(horizon_s), 0.4, 2.0)
    corridor = frame.get("corridor") or [[0.3, 0.97], [0.7, 0.97], [0.545, 0.52], [0.455, 0.52]]
    base, far = _centerline(corridor)
    ox, oy = _norm_center(obj, width, height)
    mid_x = (base[0] + far[0]) / 2.0
    shift_sign = -1.0 if ox >= mid_x else 1.0
    uncertainty = float(obj.get("uncertainty") or 0.03)
    other = object_samples(obj, width, height, horizon_s, STEPS)
    scenarios = []
    for mode, title in (
        ("continue", "Continue"),
        ("brake", "Brake"),
        ("evade", "Evade"),
    ):
        path = _cyclist_path(base, far, mode, delay_s, brake, horizon_s, shift_sign, STEPS)
        conflict = _conflict(path, other, uncertainty)
        if conflict["level"] == "overlap":
            tone = "red"
        elif conflict["level"] == "near":
            tone = "amber"
        else:
            tone = "green"
        note = {
            "continue": "Cyclist path stays on the current corridor centerline.",
            "brake": "After the reaction delay, the path covers less of the corridor. The object keeps its observed image velocity.",
            "evade": "After the reaction delay, the path shifts away from the selected object in the image.",
        }[mode]
        scenarios.append(
            {
                "id": mode,
                "title": title,
                "path": path,
                "object_path": other,
                "tone": tone,
                "note": note,
                **conflict,
            }
        )
    return {
        "track_id": obj.get("id"),
        "display": obj.get("display") or obj.get("label"),
        "corridor_mode": frame.get("corridor_mode") or "approximate",
        "delay_s": round(delay_s, 3),
        "brake": round(brake, 3),
        "horizon_s": round(horizon_s, 3),
        "uncertainty": round(uncertainty, 4),
        "evade_direction": "left" if shift_sign < 0 else "right",
        "scenarios": scenarios,
        "disclaimer": (
            "Image-space projection only. Braking and evading move the cyclist's path, "
            "not the camera image. Overlap is not a certain collision and is not a distance in meters."
        ),
    }


def occlusion_zones(frame: dict, width: float, height: float) -> dict[str, Any]:
    zones = []
    for obj in frame.get("objects") or []:
        if obj.get("label") not in OCCLUDERS:
            continue
        box = obj.get("bbox") or [0, 0, 0, 0]
        x1, y1, x2, y2 = box[0] / width, box[1] / height, box[2] / width, box[3] / height
        bw, bh = x2 - x1, y2 - y1
        # A box has to sit low enough, and be large enough, to hide the view beyond it.
        if y2 < 0.58 or bh < 0.08 or bw < 0.05:
            continue
        outward = -1.0 if (x1 + x2) / 2.0 < 0.5 else 1.0
        top = clamp(y1 - 0.16, 0.08, y1)
        poly = [
            [round(x1, 4), round(y1, 4)],
            [round(x2, 4), round(y1, 4)],
            [round(clamp(x2 + outward * 0.07, 0.0, 1.0), 4), round(top, 4)],
            [round(clamp(x1 + outward * 0.02, 0.0, 1.0), 4), round(top, 4)],
        ]
        zones.append(
            {
                "id": obj.get("id"),
                "display": obj.get("display") or obj.get("label"),
                "label": obj.get("label"),
                "polygon": poly,
                "headline": "LIMITED VISIBILITY",
                "note": (
                    f"{obj.get('display') or 'Vehicle'} #{obj.get('id')} blocks a wedge of the view beyond its box. "
                    "Another road user could emerge from that wedge. No hidden person or vehicle is detected there."
                ),
            }
        )
    return {
        "supported": bool(zones),
        "zones": zones,
        "disclaimer": (
            "Zones are the image area hidden by a visible vehicle box. "
            "They are not detected objects."
        ),
    }


def redesign_notes(frame: dict, obj: dict | None, zones: dict) -> dict[str, Any]:
    ideas = []
    factors = (obj or {}).get("factors") or {}
    label = (obj or {}).get("label")
    risk = obj.get("risk") if obj else None
    if label == "person" and ((obj or {}).get("conflict") or (risk or 0) >= 40):
        ideas.append(
            {
                "id": "daylighting",
                "title": "Intersection daylighting",
                "why": "A pedestrian track is in or near the estimated corridor on this frame.",
                "overlay": "crossing",
            }
        )
    if zones.get("supported"):
        ideas.append(
            {
                "id": "parking",
                "title": "Keep the crossing approach clear",
                "why": "A large vehicle box hides a wedge of the forward view.",
                "overlay": "clear-corner",
            }
        )
    closing = factors.get("closing_motion") or 0
    if label in OCCLUDERS and closing >= 0.45 and (risk or 0) >= 40:
        ideas.append(
            {
                "id": "protected-lane",
                "title": "Separation from closing vehicles",
                "why": "A vehicle track is closing toward the corridor in image space.",
                "overlay": "lane",
            }
        )
    if not ideas:
        ideas.append(
            {
                "id": "none",
                "title": "No design claim for this frame",
                "why": "The tracks on this frame do not support a specific street-design note.",
                "overlay": "none",
            }
        )
    return {
        "ideas": ideas,
        "label": "Conceptual diagram only. Not a new video, and not a measured crash reduction.",
    }


def investigation_brief(
    meta: dict,
    frame: dict,
    obj: dict,
    futures: dict,
    zones: dict,
    similar: list[dict],
    redesign: dict,
) -> str:
    factors = obj.get("factors") or {}
    lines = [
        "GHOST NYC investigation. Heuristic image-space notes, not a crash finding.",
        f"Clip {meta.get('filename')} camera {meta.get('camera_id')} location label {meta.get('location')}.",
        f"Frame t={frame.get('t')}s. {obj.get('display')} #{obj.get('id')} status {obj.get('status')} score {obj.get('risk')}.",
        f"Tracking {obj.get('tracking')} hits {obj.get('hits')} confidence {obj.get('conf')}.",
        (
            "Factors: "
            f"overlap {factors.get('trajectory_overlap')}, proximity {factors.get('apparent_proximity')}, "
            f"closing {factors.get('closing_motion')}, timing {factors.get('conflict_timing')}, "
            f"certainty {factors.get('certainty')}."
        ),
        f"Corridor mode {frame.get('corridor_mode')}.",
        "Futures: "
        + "; ".join(
            f"{s['title']} image-score {s['image_score']} level {s['level']} min-distance {s['min_distance']}"
            for s in futures.get("scenarios") or []
        ),
        futures.get("disclaimer") or "",
    ]
    if zones.get("supported"):
        lines.append("Occlusion: " + " ".join(z["note"] for z in zones["zones"][:2]))
    else:
        lines.append("Occlusion: no large low vehicle box supports a blind wedge on this frame.")
    if similar:
        lines.append("Similar indexed moments (same camera and city label, not a verified street corner):")
        for hit in similar[:4]:
            lines.append(
                f"- {hit.get('filename') or hit.get('source')} t={hit.get('t_start')} score {hit.get('score')} {hit.get('caption','')[:160]}"
            )
    else:
        lines.append("Similar-moment search returned no bike-camera hits.")
    lines.append("Design notes: " + "; ".join(f"{i['title']}: {i['why']}" for i in redesign.get("ideas") or []))
    lines.append(redesign.get("label") or "")
    caption = (meta.get("reasoning_content") or "")[:400]
    if caption:
        lines.append("Segment caption, context only: " + caption)
    lines.append("The model may restate this evidence. It may not add objects, meters, or a claim that a crash would have happened.")
    return "\n".join(lines)
