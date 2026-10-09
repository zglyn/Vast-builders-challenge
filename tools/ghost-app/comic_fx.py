"""Comedic hypothetical overlays + simple SFX (fast, CPU-only).

These are playful GENERATED gags for demo storytelling — NOT crash predictions
or findings of fault. Always labeled as comedy / hypothetical.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import wave
from typing import Any

from video_edit import Frame, _idx, fill_rect, interpolate_ghost


COMIC_DISCLAIMER = (
    "COMEDIC HYPOTHETICAL — cartoon gag for storytelling only. "
    "Not observed. Not a crash prediction or finding of fault."
)


def _put_pixel(frame: Frame, x: int, y: int, color: tuple[int, int, int], a: float = 1.0) -> None:
    w, h, rgb = frame["w"], frame["h"], frame["rgb"]
    if not (0 <= x < w and 0 <= y < h):
        return
    i = _idx(w, x, y)
    if a >= 0.99:
        rgb[i], rgb[i + 1], rgb[i + 2] = color
    else:
        inv = 1.0 - a
        rgb[i] = int(rgb[i] * inv + color[0] * a)
        rgb[i + 1] = int(rgb[i + 1] * inv + color[1] * a)
        rgb[i + 2] = int(rgb[i + 2] * inv + color[2] * a)


def draw_circle(frame: Frame, cx: int, cy: int, r: int, color: tuple[int, int, int], fill: bool = True) -> None:
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                if fill or abs((x - cx) ** 2 + (y - cy) ** 2 - r * r) < r:
                    _put_pixel(frame, x, y, color)


def draw_star(frame: Frame, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    # Simple 4-point starburst
    for t in range(-r, r + 1):
        _put_pixel(frame, cx + t, cy, color)
        _put_pixel(frame, cx, cy + t, color)
        if abs(t) < r * 0.7:
            _put_pixel(frame, cx + t, cy + t, color)
            _put_pixel(frame, cx + t, cy - t, color)


def draw_text_block(
    frame: Frame,
    text: str,
    x: int,
    y: int,
    *,
    color: tuple[int, int, int] = (255, 220, 40),
    bg: tuple[int, int, int] | None = (20, 10, 10),
    scale: int = 2,
) -> None:
    """Tiny 5x7 bitmap font for a few comic words (A-Z, 0-9, !?)."""
    glyphs = {
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
        "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "G": ["01111", "10000", "10000", "10111", "10001", "10001", "01110"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "M": ["10001", "11011", "10101", "10001", "10001", "10001", "10001"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
        "W": ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "!": ["00100", "00100", "00100", "00100", "00100", "00000", "00100"],
        "?": ["01110", "10001", "00001", "00010", "00100", "00000", "00100"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
        "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    }
    cursor = x
    for ch in text.upper():
        g = glyphs.get(ch, glyphs["?"])
        gw, gh = 5 * scale, 7 * scale
        if bg:
            fill_rect(frame, [cursor - 1, y - 1, cursor + gw + 1, y + gh + 1], bg)
        for row, bits in enumerate(g):
            for col, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            _put_pixel(frame, cursor + col * scale + dx, y + row * scale + dy, color)
        cursor += gw + scale


def impact_moment(edit_spec: dict[str, Any]) -> tuple[float, float, float] | None:
    """Approximate when/where two ghost paths get closest (image-space)."""
    tracks = edit_spec.get("tracks") or []
    if len(tracks) < 2:
        # single vehicle: gag at mid horizon
        tr = tracks[0]
        path = tr["ghost_path"]
        mid_t = (path[0]["t"] + path[-1]["t"]) / 2
        cx, cy = interpolate_ghost(path, mid_t)
        return mid_t, cx, cy
    a, b = tracks[0], tracks[1]
    t0 = max(a["ghost_path"][0]["t"], b["ghost_path"][0]["t"])
    t1 = min(a["ghost_path"][-1]["t"], b["ghost_path"][-1]["t"])
    if t1 <= t0:
        return None
    best = None
    steps = 16
    for i in range(steps + 1):
        t = t0 + (t1 - t0) * i / steps
        ax, ay = interpolate_ghost(a["ghost_path"], t)
        bx, by = interpolate_ghost(b["ghost_path"], t)
        d = math.hypot(ax - bx, ay - by)
        if best is None or d < best[0]:
            best = (d, t, (ax + bx) / 2, (ay + by) / 2)
    if best is None:
        return None
    return best[1], best[2], best[3]


def pick_gag(edit_spec: dict[str, Any] | None = None, gag: str | None = None) -> str:
    """Map scenario intervention / prompt to a comic gag style."""
    if gag:
        return gag
    sc = (edit_spec or {}).get("scenario") or {}
    intervention = str(sc.get("intervention") or "")
    rationale = str(sc.get("rationale") or "").lower()
    if intervention in ("delayed_lane_change", "aborted_lane_change") or "swerve" in rationale:
        return "swerve"
    if intervention in ("stronger_braking", "earlier_braking", "later_braking"):
        return "crash"
    return "crash"


def apply_comic_overlays(
    frames: list[tuple[float, Frame]],
    edit_spec: dict[str, Any],
    *,
    gag: str | None = None,
) -> dict[str, Any]:
    """Draw comedic gags onto generated frames in-place. Returns meta for report/SFX."""
    gag = pick_gag(edit_spec, gag)
    hit = impact_moment(edit_spec)
    if hit is None:
        return {"gag": gag, "impact_t": None}
    impact_t, ix, iy = hit
    for t, fr in frames:
        # persistent top banner
        fill_rect(fr, [0, 0, fr["w"], 22], (90, 20, 20))
        draw_text_block(fr, "COMEDY WHAT-IF", 8, 4, color=(255, 200, 60), bg=None, scale=2)
        if t + 1e-6 < impact_t - 0.15:
            # pre-impact skid / swerve lines near first target
            tr = edit_spec["tracks"][0]
            cx, cy = interpolate_ghost(tr["ghost_path"], t)
            for k in range(3):
                fill_rect(fr, [cx - 30 - k * 8, cy + 8, cx - 10 - k * 8, cy + 10], (220, 220, 220))
            if gag == "swerve":
                # zig-zag tire marks
                for k in range(4):
                    ox = int(cx) - 40 + k * 12
                    oy = int(cy) + (8 if k % 2 == 0 else -8)
                    fill_rect(fr, [ox, oy, ox + 10, oy + 3], (240, 240, 240))
                if t >= impact_t - 0.35:
                    draw_text_block(fr, "SWERVE!", int(cx) - 48, int(cy) - 44, color=(120, 220, 255), scale=2)
            elif t >= impact_t - 0.35:
                draw_text_block(fr, "SKRRRT", int(cx) - 40, int(cy) - 40, color=(255, 255, 80), scale=2)
        elif abs(t - impact_t) <= 0.12:
            # boom / whoops — ring flash only (don't wash out the vehicle body)
            flash = min(1.0, 1.2 - abs(t - impact_t) * 8)
            ix_i, iy_i = int(ix), int(iy)
            outer2, inner2 = 95 * 95, 48 * 48
            for y in range(max(0, iy_i - 95), min(fr["h"], iy_i + 96)):
                for x in range(max(0, ix_i - 95), min(fr["w"], ix_i + 96), 2):
                    d2 = (x - ix) ** 2 + (y - iy) ** 2
                    if inner2 < d2 < outer2:
                        _put_pixel(fr, x, y, (255, 230, 80), a=0.22 * flash)
            draw_star(fr, ix_i + 22, iy_i - 28, 18, (255, 240, 40))
            draw_star(fr, ix_i - 26, iy_i + 16, 12, (255, 120, 40))
            draw_circle(fr, ix_i, iy_i, 36, (255, 80, 40), fill=False)
            word = "WHOOPS!" if gag == "swerve" else "BOOM!"
            draw_text_block(fr, word, ix_i - 36, max(28, iy_i - 70), color=(255, 40, 40), scale=3)
        else:
            # aftermath smoke puffs
            for k in range(4):
                draw_circle(
                    fr,
                    int(ix) + k * 10 - 15,
                    int(iy) - 10 - k * 4,
                    6 + k,
                    (180, 180, 180),
                    fill=True,
                )
            draw_text_block(fr, "HYPOTHETICAL GAG", 8, fr["h"] - 24, color=(255, 180, 80), scale=2)
    return {
        "gag": gag,
        "impact_t": impact_t,
        "impact_xy": [ix, iy],
        "disclaimer": COMIC_DISCLAIMER,
    }


def _write_tone_wav(path: str, *, kind: str = "crash", duration: float = 1.5, sr: int = 22050) -> str:
    """Synthesize a tiny comic SFX wav (no asset files)."""
    n = int(duration * sr)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            t = i / sr
            if kind == "skid":
                # descending noisy whistle
                freq = 900 - 400 * (t / max(duration, 1e-3))
                amp = 0.25 * (1.0 - t / duration)
                val = amp * math.sin(2 * math.pi * freq * t) * (1 if (i // 40) % 2 == 0 else 0.6)
            else:
                # crash: thump + noise burst
                if t < 0.08:
                    val = 0.7 * math.sin(2 * math.pi * 60 * t) * (1 - t / 0.08)
                elif t < 0.35:
                    # pseudo noise
                    val = 0.35 * (((i * 1103515245 + 12345) & 0x7FFF) / 0x7FFF - 0.5) * (0.35 - (t - 0.08)) / 0.27
                else:
                    val = 0.0
            sample = max(-1.0, min(1.0, val))
            w.writeframes(struct.pack("<h", int(sample * 30000)))
    return path


def mux_comic_audio(video_path: str, out_path: str, *, impact_t: float | None, duration: float) -> str:
    """Mux synthetic skid+crash SFX onto the export with ffmpeg (CPU)."""
    if impact_t is None:
        # just copy
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path], check=True)
        return out_path
    with tempfile.TemporaryDirectory(prefix="ghost-sfx-") as td:
        skid = os.path.join(td, "skid.wav")
        crash = os.path.join(td, "crash.wav")
        _write_tone_wav(skid, kind="skid", duration=min(0.45, max(0.2, impact_t)))
        _write_tone_wav(crash, kind="crash", duration=0.45)
        # delay crash to impact_t
        delay_ms = max(0, int(impact_t * 1000))
        # Build filter: skid from 0, crash delayed
        filter_complex = (
            f"[1]adelay=0|0[a1];"
            f"[2]adelay={delay_ms}|{delay_ms}[a2];"
            f"[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[a]"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-i",
            skid,
            "-i",
            crash,
            "-filter_complex",
            filter_complex,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            # if audio mux fails, keep silent video
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video_path, "-c", "copy", out_path],
                check=True,
            )
    return out_path
