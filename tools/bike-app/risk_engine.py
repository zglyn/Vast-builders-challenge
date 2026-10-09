"""GHOST NYC collision-risk engine.

Image-space heuristic only. Scores are normalized 0–100 interaction scores.
They are not calibrated crash probabilities, meters, or physical time-to-collision.

Tracking is a ByteTrack-style two-stage associator (high-confidence match, then
low-confidence recovery) with constant-velocity prediction. IDs live inside one
segment.

The cyan wedge is the estimated forward corridor if current image motion
continues. A weak focus-of-expansion fit falls back to a labeled approximate
corridor. Ego flow is one expansion rate about that focus, removed before
closing motion is scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

ROAD_LABELS = {
    "person": "PEDESTRIAN",
    "bicycle": "CYCLIST",
    "car": "CAR",
    "motorcycle": "MOTORCYCLE",
    "bus": "BUS",
    "truck": "TRUCK",
}

HIGH_CONF = 0.45
LOW_CONF = 0.28
IOU_HIGH = 0.18
IOU_LOW = 0.10
MAX_AGE = 8
DRAW_COAST = 2
NMS_IOU = 0.55

# Raw terms are in [0, 1]. Weights sum to 1.
# displayed = round(100 * ema(raw) * (0.55 + 0.45 * certainty))
RISK_WEIGHTS = {
    "trajectory_overlap": 0.36,
    "apparent_proximity": 0.22,
    "closing_motion": 0.24,
    "conflict_timing": 0.18,
}
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9

OVERLAP_MOTION_FLOOR = 0.45
SMOOTH_ALPHA = 0.42
MIN_HISTORY = 3
MIN_SPAN_SEC = 0.18
PREDICT_STEPS = 6
PREDICT_DT = 0.12

ASSUMPTIONS = [
    "Boxes come from the YOLO11 sidecar for this segment. No boxes are invented.",
    "The camera is uncalibrated. Proximity is image position and box scale, not meters.",
    "The bicycle is the camera. The corridor is where that view is estimated to travel.",
    "Focus of expansion is fit from tracked image motion. A weak fit is labeled approximate.",
    "Closing motion subtracts a single ego-expansion rate about that focus.",
    "Conflict timing is image-time until a projected center enters the corridor, not physical time-to-collision.",
    "Object class changes the label only. No class gets a fixed risk bonus.",
    "YOLO has no taxi class. Cars stay CAR. Parked is not a detector class; a static off-corridor box scores low because it is not closing or overlapping.",
    "Scores use an exponential moving average. Under three hits or 0.18s of span, the status is ANALYZING.",
]


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


def nms(dets: list[tuple], thr: float = NMS_IOU) -> list[tuple]:
    ordered = sorted(dets, key=lambda d: -d[2])
    keep: list[tuple] = []
    for det in ordered:
        if all(iou(det[0], k[0]) < thr for k in keep):
            keep.append(det)
    return keep


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def point_in_poly(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xinter = (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi
            if x < xinter:
                inside = not inside
        j = i
    return inside


def _dist_seg(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = clip(((px - ax) * dx + (py - ay) * dy) / den, 0.0, 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def dist_to_poly(x: float, y: float, poly: list[tuple[float, float]]) -> float:
    if point_in_poly(x, y, poly):
        return 0.0
    n = len(poly)
    return min(
        _dist_seg(x, y, poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1]) for i in range(n)
    )


def _linreg(ts: list[float], xs: list[float]) -> tuple[float, float]:
    n = len(ts)
    if n < 2:
        return 0.0, 0.0
    mt = sum(ts) / n
    mx = sum(xs) / n
    var = sum((t - mt) ** 2 for t in ts)
    if var < 1e-8:
        return 0.0, 0.0
    slope = sum((t - mt) * (x - mx) for t, x in zip(ts, xs)) / var
    resid = [x - (mx + slope * (t - mt)) for t, x in zip(ts, xs)]
    sigma = math.sqrt(sum(r * r for r in resid) / n)
    return slope, sigma


@dataclass
class Track:
    id: int
    label: str
    bbox: list[float]
    conf: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 0
    misses: int = 0
    age: int = 0
    time_since_update: int = 0
    last_t: float = 0.0
    smooth: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def predict_bbox(self, t: float) -> list[float]:
        dt = max(0.0, t - self.last_t)
        dx, dy = self.vx * dt, self.vy * dt
        x1, y1, x2, y2 = self.bbox
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def update(self, bbox: list[float], conf: float, frame_i: int, t: float, width: float, height: float) -> None:
        cx, cy = center(bbox)
        if self.history and t > self.last_t + 1e-4:
            dt = t - self.last_t
            prev = self.history[-1]
            self.vx = 0.65 * self.vx + 0.35 * ((cx - prev["cx"]) / dt)
            self.vy = 0.65 * self.vy + 0.35 * ((cy - prev["cy"]) / dt)
        area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        self.history.append(
            {
                "frame": frame_i,
                "t": t,
                "cx": cx,
                "cy": cy,
                "bottom": float(bbox[3]),
                "area": area,
                "bbox": [float(v) for v in bbox],
                "conf": float(conf),
                "nx": cx / width,
                "ny": cy / height,
                "nbottom": float(bbox[3]) / height,
            }
        )
        self.bbox = [float(v) for v in bbox]
        self.conf = float(conf)
        self.last_t = t
        self.hits += 1
        self.age += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        self.time_since_update += 1
        self.misses += 1
        self.age += 1


def _match(pool: list[Track], dets: list[tuple], t: float, iou_min: float) -> list[tuple[int, int]]:
    pairs: list[tuple[float, int, int]] = []
    for ti, tr in enumerate(pool):
        pred = tr.predict_bbox(t)
        pcx, pcy = center(pred)
        diag = math.hypot(max(8.0, pred[2] - pred[0]), max(8.0, pred[3] - pred[1]))
        dt = max(1.0 / 60.0, t - tr.last_t)
        gate = max(diag * 0.95, math.hypot(tr.vx, tr.vy) * dt * 2.4 + 28.0)
        for di, (bb, _lab, _cf) in enumerate(dets):
            cx, cy = center(bb)
            dist = math.hypot(cx - pcx, cy - pcy)
            if dist > gate:
                continue
            overlap = iou(pred, bb)
            if overlap < iou_min and dist > diag * 0.85:
                continue
            pairs.append((overlap + (1.0 - dist / gate) * 0.2, ti, di))
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    chosen: list[tuple[int, int]] = []
    for _score, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        chosen.append((ti, di))
    return chosen


def track_frames(frames: list[dict], width: float, height: float) -> tuple[list[Track], list[list[dict]]]:
    live: list[Track] = []
    done: list[Track] = []
    snapshots: list[list[dict]] = []
    next_id = 1

    for fr in frames:
        fi = int(fr.get("frame_index", 0))
        t = float(fr.get("time_sec", 0.0))
        raw = []
        for det in fr.get("detections") or []:
            label = str(det.get("label") or "")
            if label not in ROAD_LABELS:
                continue
            conf = float(det.get("confidence") or 0.0)
            if conf < LOW_CONF or "bbox" not in det:
                continue
            raw.append(([float(v) for v in det["bbox"]], label, conf))
        raw = nms(raw)
        high = [d for d in raw if d[2] >= HIGH_CONF]
        low = [d for d in raw if d[2] < HIGH_CONF]

        matched_ids: set[int] = set()
        for ti, di in _match(live, high, t, IOU_HIGH):
            bb, _lab, cf = high[di]
            live[ti].update(bb, cf, fi, t, width, height)
            matched_ids.add(live[ti].id)
        remain = [tr for tr in live if tr.id not in matched_ids]
        for ti, di in _match(remain, low, t, IOU_LOW):
            bb, _lab, cf = low[di]
            remain[ti].update(bb, cf, fi, t, width, height)
            matched_ids.add(remain[ti].id)

        for tr in live:
            if tr.id not in matched_ids:
                tr.mark_missed()

        taken_centers = []
        for tr in live:
            if tr.last_t == t:
                taken_centers.append(center(tr.bbox))
        for bb, lab, cf in high:
            cx, cy = center(bb)
            if any(math.hypot(cx - tx, cy - ty) < 10 for tx, ty in taken_centers):
                continue
            tr = Track(id=next_id, label=lab, bbox=list(bb), conf=cf)
            next_id += 1
            tr.update(bb, cf, fi, t, width, height)
            live.append(tr)

        still: list[Track] = []
        for tr in live:
            if tr.time_since_update > MAX_AGE:
                done.append(tr)
            else:
                still.append(tr)
        live = still

        snap = []
        for tr in live:
            if tr.time_since_update > DRAW_COAST:
                continue
            box = tr.bbox if tr.time_since_update == 0 else tr.predict_bbox(t)
            snap.append(
                {
                    "id": tr.id,
                    "label": tr.label,
                    "display": ROAD_LABELS.get(tr.label, tr.label.upper()),
                    "bbox": [round(float(v), 1) for v in box],
                    "conf": round(tr.conf, 3),
                    "hits": tr.hits,
                    "misses": tr.misses,
                    "time_since_update": tr.time_since_update,
                    "tracking": "coasting" if tr.time_since_update else "locked",
                    "t": round(t, 4),
                }
            )
        snapshots.append(snap)

    done.extend(live)
    return done, snapshots


def estimate_foe(samples: list[tuple[float, float, float, float]]) -> tuple[float, float] | None:
    """Fit focus of expansion. samples are (nx, ny, vnx, vny) per second."""
    rows = []
    for px, py, vx, vy in samples:
        speed = math.hypot(vx, vy)
        if speed < 0.03:
            continue
        # fx*vy - fy*vx = px*vy - py*vx
        rows.append((vy, -vx, px * vy - py * vx, min(speed, 1.5)))
    if len(rows) < 4:
        return None
    a00 = a01 = a11 = b0 = b1 = 0.0
    for vy, nvx, rhs, w in rows:
        a00 += w * vy * vy
        a01 += w * vy * nvx
        a11 += w * nvx * nvx
        b0 += w * vy * rhs
        b1 += w * nvx * rhs
    det = a00 * a11 - a01 * a01
    if abs(det) < 1e-8:
        return None
    fx = (b0 * a11 - a01 * b1) / det
    fy = (a00 * b1 - a01 * b0) / det
    if not (0.12 <= fx <= 0.88 and 0.18 <= fy <= 0.72):
        return None
    err = 0.0
    for vy, nvx, rhs, _w in rows:
        err += abs(fx * vy + fy * nvx - rhs)
    if err / len(rows) > 0.12:
        return None
    return fx, fy


def expansion_scale(samples: list[tuple[float, float, float, float]], foe: tuple[float, float]) -> float:
    fx, fy = foe
    vals = []
    for px, py, vx, vy in samples:
        dx, dy = px - fx, py - fy
        n2 = dx * dx + dy * dy
        if n2 < 2e-4:
            continue
        vals.append((vx * dx + vy * dy) / n2)
    if len(vals) < 3:
        return 0.0
    vals.sort()
    return vals[len(vals) // 2]


def corridor_polygon(foe_x: float, foe_y: float) -> list[tuple[float, float]]:
    far_y = clip(foe_y + 0.06, 0.34, 0.62)
    return [
        (clip(foe_x - 0.20, 0.02, 0.98), 0.97),
        (clip(foe_x + 0.20, 0.02, 0.98), 0.97),
        (clip(foe_x + 0.045, 0.02, 0.98), far_y),
        (clip(foe_x - 0.045, 0.02, 0.98), far_y),
    ]


def _motion(track: Track, t_max: float, width: float, height: float) -> dict[str, float] | None:
    del width, height
    hist = [h for h in track.history if h["t"] <= t_max + 1e-4]
    if len(hist) < 2:
        return None
    window = hist[-8:]
    ts = [h["t"] for h in window]
    if ts[-1] - ts[0] < 1e-3:
        return None
    vx, sx = _linreg(ts, [h["nx"] for h in window])
    vy, sy = _linreg(ts, [h["ny"] for h in window])
    bv, _sb = _linreg(ts, [h["nbottom"] for h in window])
    areas = [math.log(max(h["area"], 1.0)) for h in window]
    grow, _sg = _linreg(ts, areas)
    return {
        "nx": window[-1]["nx"],
        "ny": window[-1]["ny"],
        "nbottom": window[-1]["nbottom"],
        "vx": vx,
        "vy": vy,
        "sigma": math.hypot(sx, sy),
        "bottom_v": bv,
        "log_area_rate": grow,
        "span": ts[-1] - ts[0],
        "conf": sum(h["conf"] for h in window) / len(window),
    }


def _score_terms(
    motion: dict[str, float],
    poly: list[tuple[float, float]],
    foe: tuple[float, float],
    scale: float,
    hits: int,
    misses: int,
    coasting: bool,
) -> dict[str, Any]:
    fx, fy = foe
    nx, ny = motion["nx"], motion["ny"]
    dx, dy = nx - fx, ny - fy
    evx, evy = scale * dx, scale * dy
    rvx, rvy = motion["vx"] - evx, motion["bottom_v"] - scale * (motion["nbottom"] - fy)
    radial = 0.62 * clip(rvy / 0.45, 0.0, 1.0) + 0.38 * clip(motion["log_area_rate"] / 0.7, 0.0, 1.0)
    radial = clip(radial, 0.0, 1.0)

    horizon = fy
    ground = clip((motion["nbottom"] - horizon) / (0.98 - horizon + 1e-6), 0.0, 1.0)
    # Apparent size counts only together with a low contact point, so a large
    # box near the horizon is not treated as close.
    apparent_size = clip((motion["nbottom"] - ny) * 2.2 / 0.28, 0.0, 1.0)
    proximity = clip(0.75 * ground + 0.25 * ground * apparent_size, 0.0, 1.0)

    dists = []
    preds = []
    uncertainty = clip(motion["sigma"] * 4.0 + (0.03 if coasting else 0.0), 0.0, 0.18)
    for step in range(PREDICT_STEPS + 1):
        dt = step * PREDICT_DT
        px = clip(nx + motion["vx"] * dt, 0.0, 1.0)
        py = clip(ny + motion["vy"] * dt, 0.0, 1.0)
        preds.append((round(px, 4), round(py, 4)))
        dists.append(dist_to_poly(px, py, poly))
    # Approach rate: how fast image distance to the corridor is falling.
    if len(dists) >= 2:
        approach = clip((dists[0] - dists[-1]) / (PREDICT_STEPS * PREDICT_DT) / 0.35, 0.0, 1.0)
    else:
        approach = 0.0
    closing = max(radial, approach)

    overlap_vals = []
    entry_dt = None
    for step, dist in enumerate(dists):
        soft = 1.0 if dist <= uncertainty else math.exp(-(dist - uncertainty) / 0.07)
        overlap_vals.append(soft)
        if entry_dt is None and dist <= uncertainty:
            entry_dt = step * PREDICT_DT
    overlap = max(overlap_vals) if overlap_vals else 0.0
    inside = dists[0] <= uncertainty
    if inside and closing < 0.25:
        timing = 0.25 * proximity
    elif entry_dt is not None:
        timing = math.exp(-entry_dt / 0.55)
    else:
        timing = 0.0

    overlap_term = overlap * (OVERLAP_MOTION_FLOOR + (1.0 - OVERLAP_MOTION_FLOOR) * closing)
    prox_eff = proximity * (0.12 + 0.88 * max(overlap, closing))
    raw = (
        RISK_WEIGHTS["trajectory_overlap"] * overlap_term
        + RISK_WEIGHTS["apparent_proximity"] * prox_eff
        + RISK_WEIGHTS["closing_motion"] * closing
        + RISK_WEIGHTS["conflict_timing"] * timing
    )
    hit_ratio = hits / max(1, hits + misses)
    age = clip(hits / 6.0, 0.0, 1.0)
    certainty = clip(0.45 * motion["conf"] + 0.35 * hit_ratio + 0.20 * age, 0.0, 1.0)
    if coasting:
        certainty *= 0.72
    analyzing = hits < MIN_HISTORY or motion["span"] < MIN_SPAN_SEC
    conflict = overlap >= 0.45 and closing >= 0.35
    return {
        "raw": raw,
        "certainty": certainty,
        "analyzing": analyzing,
        "overlap": overlap,
        "overlap_term": overlap_term,
        "proximity": proximity,
        "proximity_effective": prox_eff,
        "closing": closing,
        "timing": timing,
        "conflict": conflict,
        "trajectory": preds,
        "uncertainty": uncertainty,
        "entry_dt": entry_dt,
        "radial": radial,
        "approach": approach,
    }


def _band(score: float | None) -> str:
    if score is None:
        return "analyzing"
    if score < 25:
        return "green"
    if score < 50:
        return "yellow"
    if score < 75:
        return "orange"
    return "red"


def analyze_detections(payload: dict) -> dict[str, Any]:
    frames = payload.get("frames") or []
    if not frames:
        return {
            "fps": payload.get("fps") or 30,
            "width": 0,
            "height": 0,
            "frame_count": 0,
            "detection_count": 0,
            "tracked_objects": 0,
            "weights": RISK_WEIGHTS,
            "assumptions": ASSUMPTIONS,
            "frames": [],
            "tracks": [],
            "timeline": [],
            "highlight": None,
            "corridor_mode": "unavailable",
            "classes_seen": [],
        }
    shape = frames[0].get("shape") or [1080, 1920]
    height = float(shape[0])
    width = float(shape[1])
    fps = float(payload.get("fps") or 30.0)
    tracks, snapshots = track_frames(frames, width, height)
    by_id = {tr.id: tr for tr in tracks}

    classes: set[str] = set()
    out_frames = []
    series: dict[int, list[dict]] = {tr.id: [] for tr in tracks}
    highlight = {"t": 0.0, "score": -1.0, "id": None}

    for fr, snap in zip(frames, snapshots):
        t = float(fr.get("time_sec", 0.0))
        samples = []
        motions = {}
        for item in snap:
            tr = by_id[item["id"]]
            motion = _motion(tr, t, width, height)
            if motion and item["hits"] >= 3:
                motions[tr.id] = motion
                samples.append((motion["nx"], motion["ny"], motion["vx"], motion["vy"]))
        foe = estimate_foe(samples)
        mode = "estimated"
        if foe is None:
            mode = "approximate"
            # A gentle shared lateral drift can nudge the wedge. One fast object
            # must not drag the corridor off the forward axis.
            shift_samples = sorted(m["vx"] for m in motions.values() if m["ny"] > 0.55)
            shift = 0.0
            if len(shift_samples) >= 4:
                med = shift_samples[len(shift_samples) // 2]
                if abs(med) < 0.22:
                    shift = clip(-med * 1.1, -0.08, 0.08)
            foe = (clip(0.50 + shift, 0.2, 0.8), 0.46)
        scale = expansion_scale(samples, foe) if mode == "estimated" else 0.0
        poly = corridor_polygon(*foe)
        objects = []
        max_risk = None
        high_count = 0
        conflict_count = 0
        for item in snap:
            tr = by_id[item["id"]]
            classes.add(item["display"])
            motion = motions.get(tr.id) or _motion(tr, t, width, height)
            base = {
                **item,
                "risk": None,
                "band": "analyzing",
                "status": "ANALYZING",
                "conflict": False,
                "trajectory": [],
                "uncertainty": 0.0,
                "factors": None,
            }
            if motion is None:
                objects.append(base)
                continue
            terms = _score_terms(
                motion,
                poly,
                foe,
                scale,
                int(item["hits"]),
                int(item["misses"]),
                item["tracking"] == "coasting",
            )
            if terms["analyzing"]:
                objects.append({**base, "trajectory": terms["trajectory"], "uncertainty": round(terms["uncertainty"], 4)})
                continue
            measured = 100.0 * terms["raw"] * (0.55 + 0.45 * terms["certainty"])
            if tr.smooth is None:
                tr.smooth = measured
            else:
                tr.smooth = SMOOTH_ALPHA * measured + (1.0 - SMOOTH_ALPHA) * tr.smooth
            score = round(clip(tr.smooth, 0.0, 100.0), 1)
            band = _band(score)
            if band in ("orange", "red"):
                high_count += 1
            if terms["conflict"]:
                conflict_count += 1
            if max_risk is None or score > max_risk:
                max_risk = score
            if score > highlight["score"]:
                highlight = {"t": round(t, 3), "score": score, "id": tr.id, "label": item["display"]}
            factors = {
                "trajectory_overlap": round(terms["overlap_term"], 3),
                "apparent_proximity": round(terms["proximity_effective"], 3),
                "closing_motion": round(terms["closing"], 3),
                "conflict_timing": round(terms["timing"], 3),
                "certainty": round(terms["certainty"], 3),
                "image_conflict_sec": None if terms["entry_dt"] is None else round(terms["entry_dt"], 3),
            }
            obj = {
                **item,
                "risk": score,
                "band": band,
                "status": "SCORED",
                "conflict": terms["conflict"],
                "trajectory": terms["trajectory"],
                "uncertainty": round(terms["uncertainty"], 4),
                "factors": factors,
            }
            objects.append(obj)
            series[tr.id].append(
                {
                    "t": round(t, 3),
                    "risk": score,
                    "overlap": factors["trajectory_overlap"],
                    "proximity": factors["apparent_proximity"],
                    "closing": factors["closing_motion"],
                    "timing": factors["conflict_timing"],
                }
            )
        out_frames.append(
            {
                "t": round(t, 4),
                "corridor": [[round(x, 4), round(y, 4)] for x, y in poly],
                "corridor_mode": mode,
                "foe": [round(foe[0], 4), round(foe[1], 4)],
                "objects": objects,
                "max_risk": max_risk,
                "high_count": high_count,
                "conflict_count": conflict_count,
                "tracked": len(objects),
            }
        )

    track_rows = []
    for tr in sorted(tracks, key=lambda t: t.id):
        rows = series.get(tr.id) or []
        scored = [r["risk"] for r in rows]
        track_rows.append(
            {
                "id": tr.id,
                "label": tr.label,
                "display": ROAD_LABELS.get(tr.label, tr.label.upper()),
                "hits": tr.hits,
                "misses": tr.misses,
                "max_risk": max(scored) if scored else None,
                "t_start": round(tr.history[0]["t"], 3) if tr.history else None,
                "t_end": round(tr.history[-1]["t"], 3) if tr.history else None,
                "series": rows[:: max(1, len(rows) // 40)],
            }
        )
    det_count = sum(len(fr.get("detections") or []) for fr in frames)
    if highlight["id"] is None:
        highlight = None
    return {
        "fps": fps,
        "width": int(width),
        "height": int(height),
        "frame_count": len(frames),
        "detection_count": det_count,
        "tracked_objects": len(tracks),
        "weights": RISK_WEIGHTS,
        "assumptions": ASSUMPTIONS,
        "formula": (
            "raw = 0.36*overlap_term + 0.22*proximity_effective + 0.24*closing + 0.18*timing; "
            "overlap_term = overlap*(0.45+0.55*closing); "
            "proximity_effective = proximity*(0.12+0.88*max(overlap, closing)); "
            "score = ema(100*raw*(0.55+0.45*certainty))"
        ),
        "frames": out_frames,
        "tracks": track_rows,
        "timeline": [
            {
                "t": fr["t"],
                "max_risk": fr["max_risk"],
                "high_count": fr["high_count"],
                "conflict_count": fr["conflict_count"],
                "tracked": fr["tracked"],
            }
            for fr in out_frames[:: max(1, len(out_frames) // 120)]
        ],
        "highlight": highlight,
        "corridor_mode": out_frames[-1]["corridor_mode"] if out_frames else "unavailable",
        "classes_seen": sorted(classes),
    }


def explain_factors(obj: dict, corridor_mode: str, caption: str) -> str:
    label = obj.get("display") or obj.get("label") or "object"
    tid = obj.get("id")
    if obj.get("status") != "SCORED" or obj.get("risk") is None:
        return (
            f"{label} #{tid} does not have enough consecutive observations yet. "
            "Status is ANALYZING. No interaction score is shown."
        )
    factors = obj.get("factors") or {}
    closing = factors.get("closing_motion") or 0
    overlap = factors.get("trajectory_overlap") or 0
    if closing >= 0.55:
        motion = "its image motion is closing toward the cyclist after ego-flow removal"
    elif closing <= 0.2:
        motion = "it is not closing on the estimated corridor"
    else:
        motion = "its relative motion is mixed"
    if overlap >= 0.35:
        path = "The projected image path overlaps the estimated cycling corridor."
    else:
        path = "The projected image path stays mostly outside the estimated cycling corridor."
    timing = factors.get("image_conflict_sec")
    timing_txt = (
        f" Image-time to corridor entry is about {timing}s."
        if timing not in (None, 0)
        else " No separate physical time-to-collision is estimated."
    )
    text = (
        f"{label} #{tid} is a YOLO detection ({obj.get('label')}) with confidence {obj.get('conf')}. "
        f"Tracking is {obj.get('tracking')} after {obj.get('hits')} hits. {motion.capitalize()}. {path}{timing_txt} "
        f"Heuristic score {obj.get('risk')}% from overlap {factors.get('trajectory_overlap')}, "
        f"proximity {factors.get('apparent_proximity')}, closing {factors.get('closing_motion')}, "
        f"timing {factors.get('conflict_timing')}, certainty {factors.get('certainty')}. "
        f"Corridor mode: {corridor_mode}. "
        "This is a normalized image-space interaction score, not a crash probability."
    )
    if caption:
        text += " Segment caption is context only and is not treated as object intent."
    return text
