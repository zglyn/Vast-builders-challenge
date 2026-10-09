"""Geometric video edit: erase observed vehicle, paste real crop along ghost path.

Sprint upgrades: OpenCV Telea inpaint (with pure-Python fallback), alpha matte,
scale/rotate from ghost motion. Output is labeled GENERATED.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable

from edit_spec import validate_edit_spec

Frame = dict[str, Any]  # {w, h, rgb: bytearray}

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    _HAS_CV2 = True
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _HAS_CV2 = False


def cv2_available() -> bool:
    return bool(_HAS_CV2)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_EXPORT_DIR = os.environ.get("GHOST_EXPORT_DIR", "/tmp/ghost-exports")


def _idx(w: int, x: int, y: int) -> int:
    return (y * w + x) * 3


def make_frame(w: int, h: int, color: tuple[int, int, int] = (40, 44, 52)) -> Frame:
    r, g, b = color
    buf = bytearray([r, g, b] * (w * h))
    return {"w": w, "h": h, "rgb": buf}


def clone_frame(frame: Frame) -> Frame:
    return {"w": frame["w"], "h": frame["h"], "rgb": bytearray(frame["rgb"])}


def fill_rect(frame: Frame, bbox: list[float], color: tuple[int, int, int]) -> None:
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    r, g, b = color
    for y in range(y1, y2):
        row = _idx(w, x1, y)
        for x in range(x1, x2):
            i = row + (x - x1) * 3
            rgb[i] = r
            rgb[i + 1] = g
            rgb[i + 2] = b


def draw_rect_outline(
    frame: Frame, bbox: list[float], color: tuple[int, int, int], thickness: int = 2
) -> None:
    x1, y1, x2, y2 = bbox
    for t in range(thickness):
        fill_rect(frame, [x1 - t, y1 - t, x2 + t, y1 - t + 1], color)
        fill_rect(frame, [x1 - t, y2 + t - 1, x2 + t, y2 + t], color)
        fill_rect(frame, [x1 - t, y1 - t, x1 - t + 1, y2 + t], color)
        fill_rect(frame, [x2 + t - 1, y1 - t, x2 + t, y2 + t], color)


def _clamp_bbox(bbox: list[float], w: int, h: int) -> list[int]:
    x1 = max(0, min(w, int(bbox[0])))
    y1 = max(0, min(h, int(bbox[1])))
    x2 = max(0, min(w, int(bbox[2])))
    y2 = max(0, min(h, int(bbox[3])))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def sample_border_color(frame: Frame, bbox: list[float], pad: int = 3) -> tuple[int, int, int]:
    """Average RGB just outside the bbox — cheap CV-inpaint stand-in."""
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    x1, y1, x2, y2 = _clamp_bbox(bbox, w, h)
    xs, ys = [], []
    for x in range(max(0, x1 - pad), min(w, x2 + pad)):
        for y in (max(0, y1 - pad), min(h - 1, max(0, y2 + pad - 1))):
            xs.append(x)
            ys.append(y)
    for y in range(max(0, y1 - pad), min(h, y2 + pad)):
        for x in (max(0, x1 - pad), min(w - 1, max(0, x2 + pad - 1))):
            xs.append(x)
            ys.append(y)
    if not xs:
        return (40, 44, 52)
    sr = sg = sb = 0
    n = 0
    for x, y in zip(xs, ys):
        if not (0 <= x < w and 0 <= y < h):
            continue
        # skip interior
        if x1 <= x < x2 and y1 <= y < y2:
            continue
        i = _idx(w, x, y)
        sr += rgb[i]
        sg += rgb[i + 1]
        sb += rgb[i + 2]
        n += 1
    if n == 0:
        return (40, 44, 52)
    return (sr // n, sg // n, sb // n)


def _frame_to_bgr(frame: Frame):
    assert _HAS_CV2 and np is not None and cv2 is not None
    w, h = frame["w"], frame["h"]
    arr = np.frombuffer(frame["rgb"], dtype=np.uint8).reshape(h, w, 3)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_frame(bgr, frame: Frame) -> None:
    assert _HAS_CV2 and np is not None and cv2 is not None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    frame["rgb"][:] = rgb.reshape(-1).tobytes()


def erase_bbox(
    frame: Frame,
    bbox: list[float],
    inflate: float = 1.0,
    *,
    protect_boxes: list[list[float]] | None = None,
    tight: bool = True,
) -> None:
    """Erase vehicle region via OpenCV Telea inpaint, else multi-pass neighbor fill.

    tight=True (default): small pad, tiny dilate, small radius — avoids smearing
    nearby cars when the truck brakes close to them. GPU does not help this.
    protect_boxes: other vehicles to carve out of the erase mask.
    """
    w, h = frame["w"], frame["h"]
    x1, y1, x2, y2 = bbox
    # Slightly shrink core box so adjacent vehicles aren't eaten, then tiny pad
    if tight:
        shrink = 2.0
        core = [x1 + shrink, y1 + shrink, x2 - shrink, y2 - shrink]
        if core[2] - core[0] < 4 or core[3] - core[1] < 4:
            core = [x1, y1, x2, y2]
        pad = max(0.0, float(inflate))
        padded = _clamp_bbox([core[0] - pad, core[1] - pad, core[2] + pad, core[3] + pad], w, h)
    else:
        padded = _clamp_bbox([x1 - inflate, y1 - inflate, x2 + inflate, y2 + inflate], w, h)
    if padded[2] <= padded[0] or padded[3] <= padded[1]:
        return

    if _HAS_CV2 and np is not None and cv2 is not None:
        bgr = _frame_to_bgr(frame)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (padded[0], padded[1]), (padded[2] - 1, padded[3] - 1), 255, thickness=-1)
        if tight:
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
            radius = 2
        else:
            mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
            radius = max(3, int(0.04 * max(padded[2] - padded[0], padded[3] - padded[1])))
        # Keep neighboring vehicles sharp — punch them out of the inpaint mask
        for pb in protect_boxes or []:
            px1, py1, px2, py2 = _clamp_bbox(pb, w, h)
            if px2 > px1 and py2 > py1:
                # grow protect slightly so Telea doesn't nibble their edges
                cv2.rectangle(
                    mask,
                    (max(0, px1 - 1), max(0, py1 - 1)),
                    (min(w - 1, px2), min(h - 1, py2)),
                    0,
                    thickness=-1,
                )
        if int(mask.sum()) < 10:
            return
        filled = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
        _bgr_to_frame(filled, frame)
        return

    # Pure-Python fallback: seed with border color, then pull neighbors inward
    color = sample_border_color(frame, padded)
    fill_rect(frame, padded, color)
    rgb = frame["rgb"]
    for _ in range(4):
        snapshot = bytes(rgb)
        for y in range(padded[1], padded[3]):
            for x in range(padded[0], padded[2]):
                # only refine interior ring
                if (
                    x <= padded[0] + 1
                    or y <= padded[1] + 1
                    or x >= padded[2] - 2
                    or y >= padded[3] - 2
                ):
                    continue
                acc = [0, 0, 0]
                n = 0
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < w and 0 <= yy < h:
                        j = _idx(w, xx, yy)
                        acc[0] += snapshot[j]
                        acc[1] += snapshot[j + 1]
                        acc[2] += snapshot[j + 2]
                        n += 1
                if n:
                    i = _idx(w, x, y)
                    rgb[i] = acc[0] // n
                    rgb[i + 1] = acc[1] // n
                    rgb[i + 2] = acc[2] // n


def crop_patch(frame: Frame, bbox: list[float]) -> dict[str, Any]:
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    out = bytearray(pw * ph * 3)
    for y in range(ph):
        src = _idx(w, x1, y1 + y)
        dst = y * pw * 3
        out[dst : dst + pw * 3] = rgb[src : src + pw * 3]
    return {"w": pw, "h": ph, "rgb": out, "base_w": pw, "base_h": ph}


def build_alpha_matte(pw: int, ph: int, *, feather: float = 0.28) -> list[float]:
    """Elliptical soft matte in [0,1] so pasted vehicles blend into the road.

    Note: ellipses clip truck corners — prefer build_rect_matte for quality exports.
    """
    cx, cy = (pw - 1) / 2.0, (ph - 1) / 2.0
    rx = max(1.0, pw / 2.0)
    ry = max(1.0, ph / 2.0)
    feather = max(0.05, min(0.49, feather))
    inner = 1.0 - feather
    matte = [0.0] * (pw * ph)
    for y in range(ph):
        for x in range(pw):
            nx = (x - cx) / rx
            ny = (y - cy) / ry
            r = math.sqrt(nx * nx + ny * ny)
            if r <= inner:
                a = 1.0
            elif r >= 1.0:
                a = 0.0
            else:
                a = 1.0 - (r - inner) / (1.0 - inner)
            matte[y * pw + x] = a
    return matte


def build_rect_matte(pw: int, ph: int, *, edge: int = 4) -> list[float]:
    """Rectangular matte: opaque vehicle body, thin soft edge only.

    Keeps the whole truck/car visible (ellipse was clipping corners into the road).
    """
    edge = max(0, min(edge, min(pw, ph) // 3))
    matte = [1.0] * (pw * ph)
    if edge <= 0:
        return matte
    for y in range(ph):
        for x in range(pw):
            d = min(x, y, pw - 1 - x, ph - 1 - y)
            if d < edge:
                matte[y * pw + x] = d / float(edge)
    return matte


def ghost_motion(path: list[dict], t: float) -> tuple[float, float, float]:
    """Return (vx, vy, speed) image-space px/s near t from ghost path."""
    if len(path) < 2:
        return 0.0, 0.0, 0.0
    # find segment
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a["t"] <= t <= b["t"] or i == len(path) - 2:
            dt = (b["t"] - a["t"]) or 1e-3
            vx = (b["cx"] - a["cx"]) / dt
            vy = (b["cy"] - a["cy"]) / dt
            return vx, vy, math.hypot(vx, vy)
    a, b = path[0], path[1]
    dt = (b["t"] - a["t"]) or 1e-3
    vx = (b["cx"] - a["cx"]) / dt
    vy = (b["cy"] - a["cy"]) / dt
    return vx, vy, math.hypot(vx, vy)


def transform_patch(
    patch: dict[str, Any],
    *,
    scale: float = 1.0,
    angle_deg: float = 0.0,
) -> dict[str, Any]:
    """Scale + rotate vehicle crop (OpenCV when available)."""
    scale = max(0.35, min(2.5, float(scale)))
    angle_deg = float(angle_deg)
    pw, ph, prgb = patch["w"], patch["h"], patch["rgb"]

    if _HAS_CV2 and np is not None and cv2 is not None:
        arr = np.frombuffer(prgb, dtype=np.uint8).reshape(ph, pw, 3)
        new_w = max(1, int(round(pw * scale)))
        new_h = max(1, int(round(ph * scale)))
        interp = cv2.INTER_CUBIC if scale >= 0.95 else cv2.INTER_AREA
        scaled = cv2.resize(arr, (new_w, new_h), interpolation=interp)
        if abs(angle_deg) < 0.25:
            out = scaled
        else:
            center = (new_w / 2.0, new_h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            # expand canvas so corners aren't clipped
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            nw = int(new_h * sin + new_w * cos)
            nh = int(new_h * cos + new_w * sin)
            M[0, 2] += (nw / 2.0) - center[0]
            M[1, 2] += (nh / 2.0) - center[1]
            out = cv2.warpAffine(
                scaled,
                M,
                (nw, nh),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        return {
            "w": int(out.shape[1]),
            "h": int(out.shape[0]),
            "rgb": bytearray(out.reshape(-1).tobytes()),
            "base_w": patch.get("base_w", pw),
            "base_h": patch.get("base_h", ph),
        }

    # Pure-Python nearest-neighbor scale + no-op small angles
    tw = max(1, int(round(pw * scale)))
    th = max(1, int(round(ph * scale)))
    scaled = _resize_patch_nearest(patch, tw, th)
    if abs(angle_deg) < 1.0:
        return scaled
    # Simple rotate about center (nearest)
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    # output bounds
    corners = [(-tw / 2, -th / 2), (tw / 2, -th / 2), (-tw / 2, th / 2), (tw / 2, th / 2)]
    xs, ys = [], []
    for x, y in corners:
        xs.append(x * ca - y * sa)
        ys.append(x * sa + y * ca)
    nw = max(1, int(math.ceil(max(xs) - min(xs))))
    nh = max(1, int(math.ceil(max(ys) - min(ys))))
    out = bytearray(nw * nh * 3)
    srgb = scaled["rgb"]
    for y in range(nh):
        for x in range(nw):
            # inverse map
            dx, dy = x - nw / 2.0, y - nh / 2.0
            sx = dx * ca + dy * sa + tw / 2.0
            sy = -dx * sa + dy * ca + th / 2.0
            ix, iy = int(sx), int(sy)
            if 0 <= ix < tw and 0 <= iy < th:
                si = (iy * tw + ix) * 3
                di = (y * nw + x) * 3
                out[di : di + 3] = srgb[si : si + 3]
    return {"w": nw, "h": nh, "rgb": out, "base_w": patch.get("base_w", pw), "base_h": patch.get("base_h", ph)}


def blit_centered(
    frame: Frame,
    patch: dict[str, Any],
    cx: float,
    cy: float,
    *,
    alpha: float = 1.0,
    use_matte: bool = True,
    matte_feather: float = 0.28,
    matte_shape: str = "ellipse",
    matte_edge: int = 4,
) -> list[float]:
    """Paste patch centered at (cx, cy).

    matte_shape 'rect' keeps the full vehicle visible (quality default);
    'ellipse' soft-blends but clips truck corners.
    """
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    pw, ph, prgb = patch["w"], patch["h"], patch["rgb"]
    a0 = max(0.0, min(1.0, float(alpha)))
    if not use_matte:
        matte = [1.0] * (pw * ph)
    elif matte_shape == "rect":
        matte = build_rect_matte(pw, ph, edge=matte_edge)
    else:
        matte = build_alpha_matte(pw, ph, feather=matte_feather)
    x1 = int(round(cx - pw / 2))
    y1 = int(round(cy - ph / 2))
    for y in range(ph):
        dy = y1 + y
        if dy < 0 or dy >= h:
            continue
        for x in range(pw):
            dx = x1 + x
            if dx < 0 or dx >= w:
                continue
            wgt = a0 * matte[y * pw + x]
            if wgt <= 0.01:
                continue
            si = (y * pw + x) * 3
            # skip near-black rotation padding
            if prgb[si] + prgb[si + 1] + prgb[si + 2] < 12 and wgt < 0.95:
                continue
            di = _idx(w, dx, dy)
            if wgt >= 0.995:
                rgb[di] = prgb[si]
                rgb[di + 1] = prgb[si + 1]
                rgb[di + 2] = prgb[si + 2]
            else:
                inv = 1.0 - wgt
                rgb[di] = int(rgb[di] * inv + prgb[si] * wgt)
                rgb[di + 1] = int(rgb[di + 1] * inv + prgb[si + 1] * wgt)
                rgb[di + 2] = int(rgb[di + 2] * inv + prgb[si + 2] * wgt)
    return [float(x1), float(y1), float(x1 + pw), float(y1 + ph)]


def translate_patch(patch: dict[str, Any], dx: float, dy: float) -> dict[str, Any]:
    """Phase 3: warp patch by integer translation (ghost-motion optical-flow stand-in)."""
    # Translation of content inside a same-sized canvas with transparent=zero fill from edges
    pw, ph, prgb = patch["w"], patch["h"], patch["rgb"]
    out = bytearray(pw * ph * 3)  # black; blit alpha will feather
    ox = int(round(dx))
    oy = int(round(dy))
    for y in range(ph):
        sy = y - oy
        if sy < 0 or sy >= ph:
            continue
        for x in range(pw):
            sx = x - ox
            if sx < 0 or sx >= pw:
                continue
            si = (sy * pw + sx) * 3
            di = (y * pw + x) * 3
            out[di : di + 3] = prgb[si : si + 3]
    return {"w": pw, "h": ph, "rgb": out}


def _resize_patch_nearest(patch: dict[str, Any], tw: int, th: int) -> dict[str, Any]:
    """Nearest-neighbor resize so temporal blends tolerate edge-clipped crops."""
    sw, sh, srgb = patch["w"], patch["h"], patch["rgb"]
    if sw == tw and sh == th:
        return patch
    tw, th = max(1, tw), max(1, th)
    out = bytearray(tw * th * 3)
    for y in range(th):
        sy = min(sh - 1, int(y * sh / th))
        for x in range(tw):
            sx = min(sw - 1, int(x * sw / tw))
            si = (sy * sw + sx) * 3
            di = (y * tw + x) * 3
            out[di : di + 3] = srgb[si : si + 3]
    return {"w": tw, "h": th, "rgb": out}


def blend_patches(a: dict[str, Any], b: dict[str, Any], t: float) -> dict[str, Any]:
    """Linear blend of two patches (resized to b's size). t=0 → a, t=1 → b."""
    if a["w"] != b["w"] or a["h"] != b["h"]:
        a = _resize_patch_nearest(a, b["w"], b["h"])
    u = max(0.0, min(1.0, t))
    n = b["w"] * b["h"] * 3
    out = bytearray(n)
    ar, br = a["rgb"], b["rgb"]
    for i in range(n):
        out[i] = int(ar[i] * (1.0 - u) + br[i] * u)
    return {"w": b["w"], "h": b["h"], "rgb": out}


def interpolate_ghost(path: list[dict], t: float) -> tuple[float, float]:
    if t <= path[0]["t"]:
        return path[0]["cx"], path[0]["cy"]
    if t >= path[-1]["t"]:
        return path[-1]["cx"], path[-1]["cy"]
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        if a["t"] <= t <= b["t"]:
            u = 0.0 if b["t"] == a["t"] else (t - a["t"]) / (b["t"] - a["t"])
            return a["cx"] + u * (b["cx"] - a["cx"]), a["cy"] + u * (b["cy"] - a["cy"])
    return path[-1]["cx"], path[-1]["cy"]


def observed_bbox_at(observed: list[dict], t: float) -> list[float] | None:
    if not observed:
        return None
    best = min(observed, key=lambda p: abs(p["t"] - t))
    if abs(best["t"] - t) > 0.2:
        return None
    return list(best["bbox"])


def stamp_banner(frame: Frame, text_rows: int = 18, *, split_label: str | None = None) -> None:
    """Red banner strip at top — readable as GENERATED even without fonts."""
    fill_rect(frame, [0, 0, frame["w"], text_rows], (120, 20, 20))
    for x in range(0, frame["w"], 24):
        fill_rect(frame, [x, 4, x + 12, 12], (255, 193, 77))
    if split_label == "reality":
        fill_rect(frame, [0, 0, frame["w"] // 2, 4], (61, 255, 168))
    elif split_label == "generated":
        fill_rect(frame, [frame["w"] // 2, 0, frame["w"], 4], (255, 92, 92))


def render_geometric_edit(
    base_frames: list[tuple[float, Frame]],
    edit_spec: dict[str, Any],
    *,
    appearance_patches: dict[int, dict[str, Any]],
    temporal_polish: bool = False,
    stamp: bool = True,
    motion_orient: bool = True,
    draw_outline: bool = False,
    matte_feather: float = 0.28,
    mild_motion: bool = False,
    matte_shape: str = "ellipse",
) -> list[tuple[float, Frame]]:
    """Erase observed vehicle + paste appearance along ghost path.

    Sprint: Telea/improved erase, alpha matte, scale/rotate from ghost heading.
    temporal_polish: blend ghost-warped previous insert with fresh paste.
    matte_shape 'rect' = full vehicle visible (quality); ellipse clips corners.
    mild_motion: clamp scale/angle so braking doesn't shrink/blur the crop.
    """
    spec = validate_edit_spec(edit_spec)
    event_t = spec["event_t"]
    out: list[tuple[float, Frame]] = []
    prev_center: dict[int, tuple[float, float]] = {}
    prev_insert: dict[int, dict[str, Any]] = {}
    base_heading: dict[int, float] = {}
    base_speed: dict[int, float] = {}

    for tr in spec["tracks"]:
        tid = tr["track_id"]
        vx0, vy0, sp0 = ghost_motion(tr["ghost_path"], event_t)
        base_heading[tid] = math.degrees(math.atan2(vy0, vx0)) if sp0 > 1e-3 else 0.0
        base_speed[tid] = sp0 if sp0 > 1e-3 else 1.0

    neighbors = list(edit_spec.get("neighbor_boxes") or spec.get("neighbor_boxes") or [])

    for t, src in base_frames:
        fr = clone_frame(src)
        if t + 1e-6 >= event_t:
            # Protect nearby (non-moved) cars from Telea smear when truck brakes close
            protect: list[list[float]] = []
            for nb in neighbors:
                pob = observed_bbox_at(nb.get("observed_boxes") or [], t)
                if pob is None:
                    pob = list((nb.get("bbox") or []))
                if pob and len(pob) == 4:
                    protect.append(pob)

            for tr in spec["tracks"]:
                tid = tr["track_id"]
                ob = observed_bbox_at(tr.get("observed_boxes") or [], t)
                if ob is None:
                    if abs(t - event_t) < 0.05:
                        ob = list(tr["appearance"]["bbox"])
                if ob is not None:
                    # Other edited tracks also protected while we erase this one
                    others = [
                        list(otr["appearance"]["bbox"])
                        for otr in spec["tracks"]
                        if otr["track_id"] != tid
                    ]
                    erase_bbox(
                        fr,
                        inflate_bbox(ob, 0.02),
                        inflate=1.0,
                        protect_boxes=protect + others,
                        tight=True,
                    )
                patch = appearance_patches.get(tid)
                if patch is None:
                    continue
                cx, cy = interpolate_ghost(tr["ghost_path"], t)
                vx, vy, sp = ghost_motion(tr["ghost_path"], t)
                scale = 1.0
                angle = 0.0
                if motion_orient:
                    # Mild size change with speed (image-space heuristic)
                    scale = max(0.55, min(1.45, math.sqrt(sp / base_speed[tid])))
                    heading = math.degrees(math.atan2(vy, vx)) if sp > 1e-3 else base_heading[tid]
                    angle = heading - base_heading[tid]
                    # Clamp wild spins from noisy tracks
                    while angle > 180:
                        angle -= 360
                    while angle < -180:
                        angle += 360
                    angle = max(-35.0, min(35.0, angle))
                    if mild_motion:
                        scale = max(0.88, min(1.12, scale))
                        angle = max(-12.0, min(12.0, angle))
                use_patch = transform_patch(patch, scale=scale, angle_deg=angle)
                if temporal_polish and tid in prev_center and tid in prev_insert:
                    pcx, pcy = prev_center[tid]
                    warped = translate_patch(prev_insert[tid], cx - pcx, cy - pcy)
                    use_patch = blend_patches(warped, use_patch, 0.30)
                dest = blit_centered(
                    fr,
                    use_patch,
                    cx,
                    cy,
                    alpha=1.0,
                    use_matte=True,
                    matte_feather=matte_feather,
                    matte_shape=matte_shape,
                    matte_edge=3 if matte_shape == "rect" else 4,
                )
                if draw_outline:
                    draw_rect_outline(fr, dest, (255, 92, 92), thickness=2)
                prev_center[tid] = (cx, cy)
                prev_insert[tid] = use_patch
        if stamp:
            stamp_banner(fr)
        out.append((t, fr))
    return out


def make_split_frames(
    reality_frames: list[tuple[float, Frame]],
    generated_frames: list[tuple[float, Frame]],
) -> list[tuple[float, Frame]]:
    """Phase 4: side-by-side Reality | Generated composite."""
    n = min(len(reality_frames), len(generated_frames))
    if n == 0:
        return []
    out: list[tuple[float, Frame]] = []
    for i in range(n):
        t, left = reality_frames[i]
        _, right = generated_frames[i]
        w, h = left["w"], left["h"]
        # scale each half to half-width
        half_w = w  # keep full res per side → output 2w x h
        canvas = make_frame(half_w * 2, h, (20, 22, 28))
        # blit left (reality) and right (generated) at native size
        for y in range(h):
            src_l = _idx(w, 0, y)
            dst_l = _idx(half_w * 2, 0, y)
            canvas["rgb"][dst_l : dst_l + w * 3] = left["rgb"][src_l : src_l + w * 3]
            src_r = _idx(right["w"], 0, y)
            dst_r = _idx(half_w * 2, w, y)
            canvas["rgb"][dst_r : dst_r + w * 3] = right["rgb"][src_r : src_r + w * 3]
        # divider + labels via colored strips
        fill_rect(canvas, [w - 1, 0, w + 1, h], (255, 193, 77))
        fill_rect(canvas, [0, 0, w, 14], (20, 60, 40))
        fill_rect(canvas, [w, 0, w * 2, 14], (80, 20, 20))
        for x in range(0, w, 28):
            fill_rect(canvas, [x + 4, 3, x + 14, 10], (61, 255, 168))
            fill_rect(canvas, [w + x + 4, 3, w + x + 14, 10], (255, 92, 92))
        out.append((t, canvas))
    return out


def write_ppm(path: str, frame: Frame) -> None:
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb)


def read_ppm(path: str) -> Frame:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P6":
            raise ValueError(f"expected P6 PPM, got {magic!r}")
        # skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        parts = line.split()
        while len(parts) < 2:
            line = f.readline()
            if line.startswith(b"#"):
                continue
            parts += line.split()
        w, h = int(parts[0]), int(parts[1])
        # maxval line may still be pending if width/height shared a line with it
        if len(parts) >= 3:
            maxval = int(parts[2])
        else:
            max_line = f.readline()
            while max_line.startswith(b"#"):
                max_line = f.readline()
            maxval = int(max_line.split()[0])
        if maxval != 255:
            raise ValueError(f"unsupported PPM maxval {maxval}")
        rgb = bytearray(f.read(w * h * 3))
        if len(rgb) != w * h * 3:
            raise ValueError("truncated PPM pixel data")
    return {"w": w, "h": h, "rgb": rgb}


def scale_edit_spec(spec: dict[str, Any], sx: float, sy: float) -> dict[str, Any]:
    """Scale all image-space coords (for resized decode)."""

    def sb(b: list[float]) -> list[float]:
        return [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]

    tracks = []
    for tr in spec["tracks"]:
        app = dict(tr["appearance"])
        app["bbox"] = sb(app["bbox"])
        app["cx"] = float(app["cx"]) * sx
        app["cy"] = float(app["cy"]) * sy
        ghost = [
            {"t": p["t"], "frame": p["frame"], "cx": p["cx"] * sx, "cy": p["cy"] * sy}
            for p in tr["ghost_path"]
        ]
        observed = [
            {"t": ob["t"], "frame": ob.get("frame", 0), "bbox": sb(ob["bbox"])}
            for ob in (tr.get("observed_boxes") or [])
        ]
        tracks.append({**tr, "appearance": app, "ghost_path": ghost, "observed_boxes": observed})
    neighbors = []
    for nb in spec.get("neighbor_boxes") or []:
        nobs = [
            {"t": ob["t"], "frame": ob.get("frame", 0), "bbox": sb(ob["bbox"])}
            for ob in (nb.get("observed_boxes") or [])
        ]
        neighbors.append(
            {
                **nb,
                "bbox": sb(list(nb["bbox"])) if nb.get("bbox") else nb.get("bbox"),
                "observed_boxes": nobs,
            }
        )
    out = dict(spec)
    out["tracks"] = tracks
    out["neighbor_boxes"] = neighbors
    return out


def inflate_bbox(bbox: list[float], frac: float = 0.12) -> list[float]:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    return [x1 - bw * frac, y1 - bh * frac, x2 + bw * frac, y2 + bh * frac]


def probe_video_size(video_path: str) -> tuple[int, int]:
    """Return (width, height). Prefer OpenCV; fall back to ffprobe/ffmpeg."""
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                if w >= 2 and h >= 2:
                    return w, h
        finally:
            cap.release()
    except Exception:  # noqa: BLE001
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        w_s, h_s = out.split("x")
        return int(w_s), int(h_s)

    # Last resort: parse ffmpeg banner (static builds often ship without ffprobe).
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("cannot probe video size — need opencv, ffprobe, or ffmpeg")
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", video_path],
        capture_output=True,
        text=True,
        check=False,
    )
    blob = (proc.stderr or "") + (proc.stdout or "")
    m = re.search(r"Stream #0:\d+[^\n]*?,\s*(\d{2,5})x(\d{2,5})\b", blob)
    if not m:
        raise RuntimeError(f"cannot parse video size from ffmpeg for {video_path}")
    return int(m.group(1)), int(m.group(2))


def decode_video_frames(
    video_path: str,
    *,
    start_t: float,
    duration_sec: float,
    fps: float,
    max_width: int = 960,
) -> tuple[list[tuple[float, Frame]], float, float]:
    """Decode a short window to RGB frames via ffmpeg PPM sequence.

    Returns (frames, scale_x, scale_y) relative to native video pixels.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg required to decode real footage")
    nat_w, nat_h = probe_video_size(video_path)
    scale = 1.0 if nat_w <= max_width else max_width / float(nat_w)
    out_w = int(round(nat_w * scale))
    # keep even dims for encoders
    out_w -= out_w % 2
    out_h = int(round(nat_h * scale))
    out_h -= out_h % 2
    sx = out_w / float(nat_w)
    sy = out_h / float(nat_h)

    with tempfile.TemporaryDirectory(prefix="ghost-decode-") as td:
        pattern = os.path.join(td, "frame_%04d.ppm")
        vf = f"fps={fps},scale={out_w}:{out_h}"
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(max(0.0, start_t)),
            "-i",
            video_path,
            "-t",
            str(duration_sec),
            "-vf",
            vf,
            pattern,
        ]
        subprocess.run(cmd, check=True)
        files = sorted(f for f in os.listdir(td) if f.endswith(".ppm"))
        if not files:
            raise RuntimeError("ffmpeg produced no frames — check source/timestamps")
        frames: list[tuple[float, Frame]] = []
        for i, name in enumerate(files):
            fr = read_ppm(os.path.join(td, name))
            frames.append((start_t + i / fps, fr))
        return frames, sx, sy


def appearance_patches_from_frames(
    frames: list[tuple[float, Frame]],
    edit_spec: dict[str, Any],
    *,
    inflate: float = 0.12,
    bank: int = 3,
) -> dict[int, dict[str, Any]]:
    """Crop real vehicle pixels.

    bank=1 keeps a sharp single-frame crop (quality). bank>1 averages (softer/stabler).
    """
    if not frames:
        raise ValueError("no frames for appearance crop")
    patches: dict[int, dict[str, Any]] = {}
    for tr in edit_spec["tracks"]:
        tid = tr["track_id"]
        target_t = float(tr["appearance"].get("t", edit_spec["event_t"]))
        prior = [p for p in frames if p[0] <= target_t + 1e-6]
        if not prior:
            prior = frames[:1]
        sample = prior[-bank:] if len(prior) >= bank else prior
        bbox = inflate_bbox(tr["appearance"]["bbox"], inflate)
        crops = [crop_patch(fr, bbox) for _t, fr in sample]
        # resize all to first crop size and average
        base = crops[0]
        tw, th = base["w"], base["h"]
        if bank <= 1 or len(crops) == 1:
            patches[tid] = {
                "w": tw,
                "h": th,
                "rgb": bytearray(base["rgb"]),
                "base_w": tw,
                "base_h": th,
            }
        else:
            acc = [0] * (tw * th * 3)
            for c in crops:
                r = c if (c["w"] == tw and c["h"] == th) else _resize_patch_nearest(c, tw, th)
                for i, v in enumerate(r["rgb"]):
                    acc[i] += v
            n = len(crops)
            averaged = bytearray(acc[i] // n for i in range(len(acc)))
            patches[tid] = {"w": tw, "h": th, "rgb": averaged, "base_w": tw, "base_h": th}
        if tw < 2 or th < 2:
            raise ValueError(f"appearance crop empty for track {tid}")
    return patches


def load_real_footage(
    edit_spec: dict[str, Any],
    video_path: str,
    *,
    max_width: int | None = None,
    quality_mode: bool = False,
) -> tuple[list[tuple[float, Frame]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Load real road frames + real vehicle crops for geometric move export.

    Returns (base_frames, appearance_patches, scaled_spec).
    """
    spec = validate_edit_spec(edit_spec)
    max_width = int(max_width or os.environ.get("GHOST_EXPORT_MAX_WIDTH", "640"))
    frames, sx, sy = decode_video_frames(
        video_path,
        start_t=float(spec["event_t"]),
        duration_sec=float(spec["duration_sec"]),
        fps=float(spec["fps"]),
        max_width=max_width,
    )
    scaled = scale_edit_spec(spec, sx, sy)
    # Ensure observed boxes cover the export window (from scaled track data)
    if not any(tr.get("observed_boxes") for tr in scaled["tracks"]):
        # leave empty — erase will use appearance at event only; still OK
        pass
    # Quality: single sharp crop; modest inflate so we don't swallow neighboring cars
    patches = appearance_patches_from_frames(
        frames,
        scaled,
        inflate=0.08 if quality_mode else 0.10,
        bank=1 if quality_mode else 3,
    )
    return frames, patches, scaled


def frames_to_mp4(
    frames: list[tuple[float, Frame]],
    out_path: str,
    *,
    fps: float,
) -> str:
    if not frames:
        raise ValueError("no frames to encode")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH — required for export")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ghost-edit-") as td:
        for i, (_t, fr) in enumerate(frames):
            write_ppm(os.path.join(td, f"frame_{i:04d}.ppm"), fr)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(td, "frame_%04d.ppm"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    return out_path


def synthesize_demo_frames(
    edit_spec: dict[str, Any],
    *,
    width: int = 320,
    height: int = 180,
    road_color: tuple[int, int, int] = (48, 52, 60),
    vehicle_colors: dict[int, tuple[int, int, int]] | None = None,
) -> tuple[list[tuple[float, Frame]], dict[int, dict[str, Any]]]:
    """Build synthetic clip for tests / offline demo without VSS video."""
    spec = validate_edit_spec(edit_spec)
    fps = spec["fps"]
    event_t = spec["event_t"]
    duration = spec["duration_sec"]
    n = max(2, int(round(duration * fps)))
    vehicle_colors = vehicle_colors or {}
    frames: list[tuple[float, Frame]] = []
    patches: dict[int, dict[str, Any]] = {}

    # Appearance patches from appearance bboxes (drawn once on a blank frame)
    for tr in spec["tracks"]:
        tid = tr["track_id"]
        color = vehicle_colors.get(tid, (80, 160, 220) if tid % 2 == 0 else (220, 160, 80))
        canvas = make_frame(width, height, road_color)
        fill_rect(canvas, tr["appearance"]["bbox"], color)
        patches[tid] = crop_patch(canvas, tr["appearance"]["bbox"])

    for i in range(n):
        t = event_t + i / fps
        fr = make_frame(width, height, road_color)
        # draw "observed" vehicles continuing on observed boxes / constant from appearance
        for tr in spec["tracks"]:
            tid = tr["track_id"]
            color = vehicle_colors.get(tid, (80, 160, 220) if tid % 2 == 0 else (220, 160, 80))
            ob = observed_bbox_at(tr.get("observed_boxes") or [], t)
            if ob is None:
                # synthesize observed motion: stay at appearance center (tests often short)
                b = tr["appearance"]["bbox"]
                bw, bh = b[2] - b[0], b[3] - b[1]
                # drift slightly right to differ from ghost
                ox = tr["appearance"]["cx"] + (t - event_t) * 30.0
                oy = tr["appearance"]["cy"]
                ob = [ox - bw / 2, oy - bh / 2, ox + bw / 2, oy + bh / 2]
                # also record for erase path consistency when caller didn't supply boxes
            fill_rect(fr, ob, color)
        frames.append((t, fr))
    return frames, patches


def _fit_spec_to_canvas(spec: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    """Remap appearance/ghost/observed coords into the synthetic canvas."""
    xs: list[float] = []
    ys: list[float] = []
    for tr in spec["tracks"]:
        b = tr["appearance"]["bbox"]
        xs += [b[0], b[2], tr["appearance"]["cx"]]
        ys += [b[1], b[3], tr["appearance"]["cy"]]
        for p in tr["ghost_path"]:
            xs.append(p["cx"])
            ys.append(p["cy"])
        for ob in tr.get("observed_boxes") or []:
            bb = ob["bbox"]
            xs += [bb[0], bb[2]]
            ys += [bb[1], bb[3]]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    margin = 20.0
    sx = (width - 2 * margin) / span_x
    sy = (height - 2 * margin) / span_y
    scale = min(sx, sy)

    def tx(x: float) -> float:
        return margin + (x - min_x) * scale

    def ty(y: float) -> float:
        return margin + (y - min_y) * scale

    out_tracks = []
    for tr in spec["tracks"]:
        b = tr["appearance"]["bbox"]
        appearance = {
            **tr["appearance"],
            "bbox": [tx(b[0]), ty(b[1]), tx(b[2]), ty(b[3])],
            "cx": tx(tr["appearance"]["cx"]),
            "cy": ty(tr["appearance"]["cy"]),
        }
        ghost = [{"t": p["t"], "frame": p["frame"], "cx": tx(p["cx"]), "cy": ty(p["cy"])} for p in tr["ghost_path"]]
        observed = []
        for ob in tr.get("observed_boxes") or []:
            bb = ob["bbox"]
            observed.append(
                {
                    "t": ob["t"],
                    "frame": ob.get("frame", 0),
                    "bbox": [tx(bb[0]), ty(bb[1]), tx(bb[2]), ty(bb[3])],
                }
            )
        out_tracks.append({**tr, "appearance": appearance, "ghost_path": ghost, "observed_boxes": observed})
    fitted = dict(spec)
    fitted["tracks"] = out_tracks
    return fitted


def _ensure_observed_boxes(spec: dict[str, Any], base: list[tuple[float, Frame]]) -> None:
    for tr in spec["tracks"]:
        if tr.get("observed_boxes"):
            continue
        b = tr["appearance"]["bbox"]
        bw, bh = b[2] - b[0], b[3] - b[1]
        boxes = []
        for t, _ in base:
            ox = tr["appearance"]["cx"] + (t - spec["event_t"]) * 30.0
            oy = tr["appearance"]["cy"]
            boxes.append(
                {
                    "t": t,
                    "frame": 0,
                    "bbox": [ox - bw / 2, oy - bh / 2, ox + bw / 2, oy + bh / 2],
                }
            )
        tr["observed_boxes"] = boxes


def mask_stats(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-track mask/appearance stats for Ghost Report (Phase 4)."""
    out = []
    for tr in spec.get("tracks") or []:
        b = tr["appearance"]["bbox"]
        area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        out.append(
            {
                "track_id": tr["track_id"],
                "label": tr.get("label"),
                "appearance_bbox": b,
                "appearance_area_px": round(area, 1),
                "ghost_points": len(tr.get("ghost_path") or []),
                "observed_boxes": len(tr.get("observed_boxes") or []),
            }
        )
    return out


def export_geometric_synthetic(
    edit_spec: dict[str, Any],
    out_path: str,
    *,
    width: int = 320,
    height: int = 180,
    temporal_polish: bool = True,
    split_screen: bool = False,
) -> dict[str, Any]:
    """End-to-end export using synthetic base frames (Phases 1/3/4)."""
    spec = _fit_spec_to_canvas(validate_edit_spec(edit_spec), width, height)
    base, patches = synthesize_demo_frames(spec, width=width, height=height)
    _ensure_observed_boxes(spec, base)

    mode_parts = ["geometric_move_synthetic"]
    edited = render_geometric_edit(
        base,
        spec,
        appearance_patches=patches,
        temporal_polish=temporal_polish,
        stamp=not split_screen,
    )
    if temporal_polish:
        mode_parts.append("temporal_polish")

    final_frames = edited
    if split_screen:
        # Reality = stamped observed base; Generated = edited
        reality = []
        for t, fr in base:
            r = clone_frame(fr)
            stamp_banner(r)
            reality.append((t, r))
        generated = []
        for t, fr in edited:
            g = clone_frame(fr)
            stamp_banner(g)
            generated.append((t, g))
        final_frames = make_split_frames(reality, generated)
        mode_parts.append("split_reality_generated")

    frames_to_mp4(final_frames, out_path, fps=spec["fps"])
    return {
        "ok": True,
        "path": out_path,
        "n_frames": len(final_frames),
        "duration_sec": spec["duration_sec"],
        "fps": spec["fps"],
        "mode": "+".join(mode_parts),
        "temporal_polish": temporal_polish,
        "split_screen": split_screen,
        "seed": spec.get("seed"),
        "run_index": spec.get("run_index"),
        "model": "geometric-paste+temporal-warp (no GPU inpaint — Phase 2 deferred)",
        "mask_stats": mask_stats(spec),
        "label": "HYPOTHETICAL",
        "disclaimer": spec["disclaimer"],
    }


def edit_spec_cache_key(spec: dict[str, Any], *, options: dict[str, Any] | None = None) -> str:
    payload = {"spec": spec, "options": options or {}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _progress(job_id: str, msg: str) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["progress"].append(msg)


def _render_pipeline(
    *,
    job_id: str,
    working_spec: dict[str, Any],
    base: list[tuple[float, Frame]],
    patches: dict[int, dict[str, Any]],
    out_path: str,
    temporal_polish: bool,
    split_screen: bool,
    footage_label: str,
    generative_refine: bool = False,
    comic_mode: bool = True,
    quality_mode: bool = False,
) -> dict[str, Any]:
    """CPU encode path. Quality = sharper matte + optional Poisson (still CPU)."""
    refine_stats: dict[str, Any] = {}
    comic_meta: dict[str, Any] = {}
    # Quality implies Poisson when real footage is available
    use_poisson = bool(generative_refine or quality_mode) and footage_label == "real_footage"
    matte_feather = 0.10 if quality_mode else 0.28
    matte_shape = "rect" if quality_mode else "ellipse"
    try:
        # Remote GPU inpaint is opt-in via GHOST_GEN_INPAINT_URL only (usually unset).
        if use_poisson:
            from gen_refine import render_generative_edit

            _progress(job_id, "stage:phase2_poisson_cpu")
            edited, refine_stats = render_generative_edit(
                base,
                working_spec,
                appearance_patches=patches,
                temporal_polish=temporal_polish,
                stamp=not split_screen,
                use_remote_inpaint=bool(os.environ.get("GHOST_GEN_INPAINT_URL")),
                mild_motion=True,
                rect_matte=True,
            )
            mode = f"{footage_label}+phase2_poisson+rect"
            model = "phase2:telea+poisson+rect-matte (CPU OpenCV)"
        else:
            _progress(job_id, "stage:fast_paste")
            edited = render_geometric_edit(
                base,
                working_spec,
                appearance_patches=patches,
                temporal_polish=temporal_polish,
                stamp=not (split_screen or comic_mode),
                motion_orient=True,
                matte_feather=matte_feather,
                mild_motion=quality_mode,
                matte_shape=matte_shape,
            )
            mode = f"{footage_label}+fast_cpu" + ("+rect" if quality_mode else "")
            model = (
                "quality-cpu:rect-matte+full-vehicle"
                if quality_mode
                else "fast-cpu:telea+ellipse-matte+motion"
            )

        if temporal_polish:
            mode += "+temporal_polish"

        if comic_mode:
            from comic_fx import COMIC_DISCLAIMER, apply_comic_overlays, pick_gag

            _progress(job_id, "stage:comic_gags")
            comic_meta = apply_comic_overlays(edited, working_spec, gag=pick_gag(working_spec))
            comic_meta["disclaimer"] = COMIC_DISCLAIMER
            mode += "+comic"

        final = edited
        if split_screen:
            _progress(job_id, "stage:split_composite")
            reality = [(t, clone_frame(fr)) for t, fr in base]
            for _, fr in reality:
                stamp_banner(fr)
            generated = [(t, clone_frame(fr)) for t, fr in edited]
            for _, fr in generated:
                stamp_banner(fr)
            final = make_split_frames(reality, generated)
            mode += "+split_reality_generated"

        _progress(job_id, "stage:encode")
        silent_path = out_path if not comic_mode else out_path + ".silent.mp4"
        frames_to_mp4(final, silent_path, fps=working_spec["fps"])
        if comic_mode:
            from comic_fx import mux_comic_audio

            _progress(job_id, "stage:comic_sfx")
            mux_comic_audio(
                silent_path,
                out_path,
                impact_t=comic_meta.get("impact_t"),
                duration=float(working_spec["duration_sec"]),
            )
            try:
                os.remove(silent_path)
            except OSError:
                pass
        return {
            "mode": mode,
            "n_frames": len(final),
            "temporal_polish": temporal_polish,
            "split_screen": split_screen,
            "comic_mode": comic_mode,
            "quality_mode": quality_mode,
            "generative_refine": use_poisson,
            "fallback_used": False,
            "footage": footage_label,
            "model": model + ("+comic-gags-sfx" if comic_mode else ""),
            "cv2": cv2_available(),
            "refine_stats": refine_stats,
            "matte_feather": matte_feather,
            "comic": comic_meta,
            "seed": working_spec.get("seed"),
            "run_index": working_spec.get("run_index"),
            "mask_stats": mask_stats(working_spec),
            "disclaimer": comic_meta.get("disclaimer")
            or working_spec.get("disclaimer"),
        }
    except Exception as polish_err:  # noqa: BLE001
        _progress(job_id, f"fallback:plain_paste after {polish_err}")
        edited = render_geometric_edit(
            base, working_spec, appearance_patches=patches, temporal_polish=False, stamp=True
        )
        frames_to_mp4(edited, out_path, fps=working_spec["fps"])
        return {
            "mode": f"{footage_label}+geometric_move_fallback",
            "n_frames": len(edited),
            "temporal_polish": False,
            "split_screen": False,
            "generative_refine": False,
            "fallback_used": True,
            "fallback_reason": str(polish_err)[:200],
            "footage": footage_label,
            "model": "geometric-paste (fallback)",
            "seed": working_spec.get("seed"),
            "run_index": working_spec.get("run_index"),
            "mask_stats": mask_stats(working_spec),
        }


def start_generate_job(
    edit_spec: dict[str, Any],
    *,
    synthetic: bool = False,
    temporal_polish: bool = False,
    split_screen: bool = False,
    generative_refine: bool = False,
    comic_mode: bool = True,
    quality_mode: bool = False,
    max_width: int | None = None,
    video_path: str | None = None,
    frame_loader: Callable[[dict], tuple[list[tuple[float, Frame]], dict[int, dict], dict]] | None = None,
) -> dict[str, Any]:
    """Queue a short CPU export job.

    quality_mode: sharper vehicle (higher res, hard matte, Poisson CPU). Not a GPU Job —
    this namespace cannot schedule nvidia.com/gpu; optional remote inpaint via env only.
    """
    spec = validate_edit_spec(edit_spec)
    if max_width is None:
        max_width = 960 if quality_mode else int(os.environ.get("GHOST_EXPORT_MAX_WIDTH", "640"))
    options = {
        "synthetic": bool(synthetic),
        "temporal_polish": temporal_polish,
        "split_screen": split_screen,
        "generative_refine": generative_refine or quality_mode,
        "comic_mode": comic_mode,
        "quality_mode": quality_mode,
        "max_width": int(max_width),
        "real_footage": (not synthetic) and (video_path is not None or frame_loader is not None),
    }
    job_id = edit_spec_cache_key(spec, options=options)
    job_id += "-syn" if synthetic else "-real"
    os.makedirs(_EXPORT_DIR, exist_ok=True)
    out_path = os.path.join(_EXPORT_DIR, f"{job_id}.mp4")

    with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing.get("status") == "done" and os.path.isfile(existing.get("path", "")):
            return {**existing, "cache_hit": True}
        _jobs[job_id] = {
            "id": job_id,
            "status": "running",
            "progress": [
                "job:start",
                (
                    f"options:real={options['real_footage']},comic={comic_mode},"
                    f"quality={quality_mode},phase2={options['generative_refine']},"
                    f"max_w={max_width},temporal={temporal_polish},split={split_screen}"
                ),
            ],
            "created_at": time.time(),
            "spec": spec,
            "options": options,
            "path": out_path,
            "cache_hit": False,
            "report": None,
        }

    def _run():
        try:
            working_spec = spec
            base = None
            patches = None
            footage_label = "synthetic"

            if not synthetic:
                try:
                    _progress(job_id, "stage:load_real_footage")
                    if frame_loader is not None:
                        loaded = frame_loader(spec)
                        if len(loaded) == 3:
                            base, patches, working_spec = loaded
                        else:
                            base, patches = loaded  # type: ignore[misc]
                            working_spec = spec
                    elif video_path:
                        base, patches, working_spec = load_real_footage(
                            spec,
                            video_path,
                            max_width=int(max_width),
                            quality_mode=quality_mode,
                        )
                    else:
                        raise RuntimeError("no video_path/frame_loader for real footage")
                    footage_label = "real_footage"
                    _progress(job_id, f"stage:real_frames:{len(base)} w={base[0][1]['w']}")
                except Exception as real_err:  # noqa: BLE001
                    # Do not silently ship cartoon boxes when the user asked for real footage.
                    _progress(job_id, f"error:real-load failure: {real_err}")
                    raise RuntimeError(
                        f"real footage load failed (refusing synthetic fallback): {real_err}"
                    ) from real_err
                else:
                    synthetic_local = False
            else:
                synthetic_local = True

            if synthetic_local or base is None or patches is None:
                _progress(job_id, "stage:synthetic_boxes")
                fitted = _fit_spec_to_canvas(spec, 320, 180)
                base, patches = synthesize_demo_frames(fitted, width=320, height=180)
                _ensure_observed_boxes(fitted, base)
                working_spec = fitted
                footage_label = "synthetic"

            result_meta = _render_pipeline(
                job_id=job_id,
                working_spec=working_spec,
                base=base,
                patches=patches,
                out_path=out_path,
                temporal_polish=temporal_polish,
                split_screen=split_screen,
                footage_label=footage_label,
                generative_refine=generative_refine or quality_mode,
                comic_mode=comic_mode,
                quality_mode=quality_mode,
            )

            report = {
                "label": "HYPOTHETICAL / GENERATED" + (" / COMEDY" if comic_mode else ""),
                "job_id": job_id,
                "mode": result_meta.get("mode"),
                "model": result_meta.get("model"),
                "footage": result_meta.get("footage"),
                "seed": result_meta.get("seed"),
                "run_index": result_meta.get("run_index"),
                "temporal_polish": result_meta.get("temporal_polish"),
                "split_screen": result_meta.get("split_screen"),
                "comic_mode": result_meta.get("comic_mode"),
                "quality_mode": result_meta.get("quality_mode"),
                "comic": result_meta.get("comic"),
                "generative_refine": result_meta.get("generative_refine"),
                "refine_stats": result_meta.get("refine_stats"),
                "fallback_used": result_meta.get("fallback_used", False),
                "mask_stats": result_meta.get("mask_stats"),
                "disclaimer": result_meta.get("disclaimer") or spec["disclaimer"],
                "scenario": spec.get("scenario"),
                "source": spec.get("source"),
            }
            with _jobs_lock:
                _jobs[job_id].update(
                    {
                        "status": "done",
                        "progress": _jobs[job_id]["progress"] + ["encode:done", "job:done"],
                        "result": result_meta,
                        "report": report,
                        "error": None,
                    }
                )
        except Exception as e:  # noqa: BLE001
            with _jobs_lock:
                _jobs[job_id].update(
                    {
                        "status": "error",
                        "error": str(e)[:400],
                        "progress": _jobs[job_id]["progress"] + [f"error:{e}"],
                    }
                )

    threading.Thread(target=_run, daemon=True).start()
    with _jobs_lock:
        return dict(_jobs[job_id])


def get_job(job_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        j = _jobs.get(job_id)
        return dict(j) if j else None
