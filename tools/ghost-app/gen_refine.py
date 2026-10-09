"""Phase 2 generative refine: Poisson seamless composite + optional remote GPU inpaint.

Cluster note: team Jobs requesting nvidia.com/gpu currently fail to schedule
(Insufficient nvidia.com/gpu). This module:

1. Uses OpenCV seamlessClone (Poisson blending) for photoreal vehicle insert.
2. Optionally calls GHOST_GEN_INPAINT_URL if organizers expose a GPU inpaint NIM/API.
3. Falls back cleanly to alpha-matte paste when OpenCV/remote is unavailable.
"""

from __future__ import annotations

import math
import os
from typing import Any

import requests

from edit_spec import validate_edit_spec
from video_edit import (
    Frame,
    _HAS_CV2,
    _bgr_to_frame,
    _frame_to_bgr,
    blend_patches,
    blit_centered,
    build_alpha_matte,
    build_rect_matte,
    clone_frame,
    cv2,
    draw_rect_outline,
    erase_bbox,
    ghost_motion,
    inflate_bbox,
    interpolate_ghost,
    np,
    observed_bbox_at,
    stamp_banner,
    transform_patch,
    translate_patch,
)


def remote_inpaint_available() -> bool:
    return bool(os.environ.get("GHOST_GEN_INPAINT_URL", "").strip())


def remote_inpaint_rgb(
    frame: Frame,
    mask_bbox: list[float],
    *,
    timeout: float = 60.0,
) -> Frame | None:
    """POST RGB frame + mask bbox to a GPU inpaint service if configured.

    Returns None on any failure (caller falls back to local Telea erase).
    """
    url = os.environ.get("GHOST_GEN_INPAINT_URL", "").strip()
    if not url or not _HAS_CV2 or cv2 is None or np is None:
        return None
    try:
        bgr = _frame_to_bgr(frame)
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return None
        headers = {}
        token = os.environ.get("GPU_BEARER_TOKEN") or os.environ.get("GHOST_GEN_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        files = {"image": ("frame.png", buf.tobytes(), "image/png")}
        data = {
            "x1": str(mask_bbox[0]),
            "y1": str(mask_bbox[1]),
            "x2": str(mask_bbox[2]),
            "y2": str(mask_bbox[3]),
            "prompt": os.environ.get(
                "GHOST_GEN_INPAINT_PROMPT",
                "empty asphalt highway road surface, photorealistic, no vehicles",
            ),
        }
        r = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
        if r.status_code >= 400:
            return None
        arr = np.frombuffer(r.content, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            return None
        out = clone_frame(frame)
        if decoded.shape[0] != frame["h"] or decoded.shape[1] != frame["w"]:
            decoded = cv2.resize(decoded, (frame["w"], frame["h"]))
        _bgr_to_frame(decoded, out)
        return out
    except Exception:  # noqa: BLE001
        return None


def seamless_paste(
    frame: Frame,
    patch: dict[str, Any],
    cx: float,
    cy: float,
    *,
    rect_matte: bool = True,
) -> tuple[list[float], str]:
    """Poisson-blend vehicle crop onto the road.

    Returns (dest_bbox, method) where method is 'seamless' or 'matte'.
    rect_matte keeps the full rectangular vehicle (ellipse was clipping trucks).
    """
    matte_kw = {"use_matte": True, "matte_shape": "rect", "matte_edge": 3} if rect_matte else {
        "use_matte": True,
        "matte_feather": 0.10,
        "matte_shape": "ellipse",
    }
    if not _HAS_CV2 or cv2 is None or np is None:
        return blit_centered(frame, patch, cx, cy, **matte_kw), "matte"

    w, h = frame["w"], frame["h"]
    pw, ph = patch["w"], patch["h"]
    x1 = int(round(cx - pw / 2))
    y1 = int(round(cy - ph / 2))
    x2, y2 = x1 + pw, y1 + ph
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        return blit_centered(frame, patch, cx, cy, **matte_kw), "matte"

    try:
        bgr = _frame_to_bgr(frame)
        patch_arr = np.frombuffer(patch["rgb"], dtype=np.uint8).reshape(ph, pw, 3)
        patch_bgr = cv2.cvtColor(patch_arr, cv2.COLOR_RGB2BGR)
        matte = (
            build_rect_matte(pw, ph, edge=3)
            if rect_matte
            else build_alpha_matte(pw, ph, feather=0.12)
        )
        mask = np.zeros((ph, pw), dtype=np.uint8)
        for y in range(ph):
            for x in range(pw):
                if matte[y * pw + x] > 0.45 and int(patch_arr[y, x].sum()) >= 12:
                    mask[y, x] = 255
        if int(mask.sum()) < 50:
            return blit_centered(frame, patch, cx, cy, **matte_kw), "matte"
        center = (int(round(cx)), int(round(cy)))
        blended = cv2.seamlessClone(patch_bgr, bgr, mask, center, cv2.NORMAL_CLONE)
        _bgr_to_frame(blended, frame)
        return [float(x1), float(y1), float(x2), float(y2)], "seamless"
    except Exception:  # noqa: BLE001
        return blit_centered(frame, patch, cx, cy, **matte_kw), "matte"


def render_generative_edit(
    base_frames: list[tuple[float, Frame]],
    edit_spec: dict[str, Any],
    *,
    appearance_patches: dict[int, dict[str, Any]],
    temporal_polish: bool = True,
    stamp: bool = True,
    motion_orient: bool = True,
    use_remote_inpaint: bool = True,
    draw_outline: bool = False,
    mild_motion: bool = True,
    rect_matte: bool = True,
) -> tuple[list[tuple[float, Frame]], dict[str, Any]]:
    """Phase 2 render: remote/Telea erase + Poisson vehicle insert along ghost path.

    Poisson seamlessClone is OpenCV CPU — not a GPU Job. Remote GPU inpaint only
    if GHOST_GEN_INPAINT_URL is set (usually unavailable in this namespace).
    """
    spec = validate_edit_spec(edit_spec)
    event_t = spec["event_t"]
    out: list[tuple[float, Frame]] = []
    prev_center: dict[int, tuple[float, float]] = {}
    prev_insert: dict[int, dict[str, Any]] = {}
    base_heading: dict[int, float] = {}
    base_speed: dict[int, float] = {}
    stats = {
        "backend": "poisson_seamless",
        "remote_inpaint_calls": 0,
        "remote_inpaint_hits": 0,
        "seamless_pastes": 0,
        "matte_fallbacks": 0,
        "cv2": bool(_HAS_CV2),
        "remote_configured": remote_inpaint_available(),
        "gpu_job_note": (
            "nvidia.com/gpu Jobs currently unschedulable in this namespace; "
            "set GHOST_GEN_INPAINT_URL for a remote GPU inpaint NIM when available."
        ),
    }

    for tr in spec["tracks"]:
        tid = tr["track_id"]
        vx0, vy0, sp0 = ghost_motion(tr["ghost_path"], event_t)
        base_heading[tid] = math.degrees(math.atan2(vy0, vx0)) if sp0 > 1e-3 else 0.0
        base_speed[tid] = sp0 if sp0 > 1e-3 else 1.0

    for t, src in base_frames:
        fr = clone_frame(src)
        if t + 1e-6 >= event_t:
            protect: list[list[float]] = []
            for nb in edit_spec.get("neighbor_boxes") or []:
                pob = observed_bbox_at(nb.get("observed_boxes") or [], t)
                if pob is None:
                    pob = list(nb.get("bbox") or [])
                if pob and len(pob) == 4:
                    protect.append(pob)

            erase_boxes = []
            for tr in spec["tracks"]:
                ob = observed_bbox_at(tr.get("observed_boxes") or [], t)
                if ob is None and abs(t - event_t) < 0.05:
                    ob = list(tr["appearance"]["bbox"])
                if ob is not None:
                    erase_boxes.append(inflate_bbox(ob, 0.02))

            for box in erase_boxes:
                used_remote = False
                if use_remote_inpaint and remote_inpaint_available():
                    stats["remote_inpaint_calls"] += 1
                    remote = remote_inpaint_rgb(fr, box)
                    if remote is not None:
                        fr = remote
                        stats["remote_inpaint_hits"] += 1
                        used_remote = True
                if not used_remote:
                    erase_bbox(
                        fr,
                        box,
                        inflate=1.0,
                        protect_boxes=protect,
                        tight=True,
                    )

            for tr in spec["tracks"]:
                tid = tr["track_id"]
                patch = appearance_patches.get(tid)
                if patch is None:
                    continue
                cx, cy = interpolate_ghost(tr["ghost_path"], t)
                vx, vy, sp = ghost_motion(tr["ghost_path"], t)
                scale, angle = 1.0, 0.0
                if motion_orient:
                    scale = max(0.55, min(1.45, math.sqrt(sp / base_speed[tid])))
                    heading = math.degrees(math.atan2(vy, vx)) if sp > 1e-3 else base_heading[tid]
                    angle = heading - base_heading[tid]
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
                    use_patch = blend_patches(warped, use_patch, 0.28)
                dest, method = seamless_paste(fr, use_patch, cx, cy, rect_matte=rect_matte)
                if method == "seamless":
                    stats["seamless_pastes"] += 1
                else:
                    stats["matte_fallbacks"] += 1
                if draw_outline:
                    draw_rect_outline(fr, dest, (255, 92, 92), thickness=2)
                prev_center[tid] = (cx, cy)
                prev_insert[tid] = use_patch
        if stamp:
            stamp_banner(fr)
        out.append((t, fr))
    return out, stats
