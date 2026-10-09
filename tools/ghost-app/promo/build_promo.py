"""Assemble the GHOST promo video with ffmpeg.

Inputs (ASSETS dir): card_*.png, shot_*.png, demo_clip.mp4, export_*.mp4,
music.wav, tts/<scene>.mp3. Timeline is driven by script.json: every scene lasts
max(min_sec, narration + tail) and scenes are joined with 0.6 s crossfades.
Music is kept low and side-chain ducked under the narration.

Usage: python3 build_promo.py script.json ASSETS_DIR OUT.mp4
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys

FPS = 25
W, H = 1920, 1080
XF = 0.6                   # crossfade seconds
TAIL = 1.4                 # breathing room after each narration line
NARR_LEAD = 0.45           # narration starts this long after the scene begins
MUSIC_GAIN = 0.36          # ~ -9 dB before ducking: clearly there, never over the voice
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
TRANSITIONS = ["fade", "smoothleft", "circleopen", "fade", "smoothup", "fade", "wipeleft", "fadeblack", "smoothleft", "fade"]
# Lab screenshot: the black <canvas> rect we blend the live clip into.
LAB_CANVAS = (281, 142, 826, 464)


def probe_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], text=True
    )
    return float(out.strip())


def mean_volume_db(path: str) -> float:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", proc.stderr)
    return float(m.group(1)) if m else -20.0


def main(script_path: str, assets: str, out_path: str) -> None:
    cfg = json.load(open(script_path))
    scenes = cfg["scenes"]
    n = len(scenes)

    # --- timeline -----------------------------------------------------------
    for s in scenes:
        s["narr_path"] = os.path.join(assets, "tts", f"{s['id']}.mp3")
        s["narr_dur"] = probe_duration(s["narr_path"])
        s["dur"] = round(max(float(s.get("min_sec", 5)), NARR_LEAD + s["narr_dur"] + TAIL), 2)
    starts = []
    t = 0.0
    for k, s in enumerate(scenes):
        starts.append(t)
        t += s["dur"] - (XF if k < n - 1 else 0)
    total = round(t, 2)
    print(f"total {total:.1f}s over {n} scenes")
    for s, st in zip(scenes, starts):
        print(f"  {s['id']:<9} start {st:6.2f}  dur {s['dur']:5.2f}  narr {s['narr_dur']:5.2f}")

    # --- inputs ---------------------------------------------------------------
    args: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning", "-stats"]
    fc: list[str] = []
    idx = 0

    def add_input(*a: str) -> int:
        nonlocal idx
        args.extend(a)
        i = idx
        idx += 1
        return i

    caption_dir = os.path.join(assets, "_captions")
    os.makedirs(caption_dir, exist_ok=True)

    for k, s in enumerate(scenes):
        dur = s["dur"]
        kind = s["kind"]
        asset = os.path.join(assets, s["asset"])
        label = f"v{k}"
        zoom_in = k % 2 == 0
        if zoom_in:
            zexpr = "min(zoom+0.00035,1.09)"
        else:
            zexpr = "if(lte(on,1),1.09,max(zoom-0.00035,1.0))"
        kb = (
            f"scale=2880:-2,zoompan=z='{zexpr}':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={W}x{H}:fps={FPS}"
        )

        if kind == "card":
            i = add_input("-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.5}", "-i", asset)
            chain = f"[{i}:v]{kb}"
        elif kind == "shot" and s["id"] == "lab":
            i = add_input("-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.5}", "-i", asset)
            j = add_input("-stream_loop", "-1", "-t", f"{dur + 0.5}", "-i", os.path.join(assets, "demo_clip.mp4"))
            x, y, cw, ch = LAB_CANVAS
            fc.append(
                f"[{j}:v]fps={FPS},scale={cw}:{ch},pad={W}:{H}:{x}:{y}:black,format=rgb24[labclip{k}]"
            )
            fc.append(f"[{i}:v]fps={FPS},scale={W}:{H},format=rgb24[labshot{k}]")
            chain = f"[labshot{k}][labclip{k}]blend=all_mode=lighten:shortest=1,{kb}"
        elif kind == "shot":
            i = add_input("-loop", "1", "-framerate", str(FPS), "-t", f"{dur + 0.5}", "-i", asset)
            pan = f"scale=2880:-2,zoompan=z=1.12:d=1:x='(iw-iw/zoom)*on/{int(FPS * dur)}':y='ih/2-(ih/zoom/2)':s={W}x{H}:fps={FPS}"
            chain = f"[{i}:v]{pan}"
        elif kind == "footage":
            i = add_input("-stream_loop", "-1", "-t", f"{dur + 0.5}", "-i", asset)
            fc.append(
                f"[{i}:v]split[fa{k}][fb{k}];"
                f"[fa{k}]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=28:6,eq=brightness=-0.18:saturation=0.8[bg{k}];"
                f"[fb{k}]scale={W}:{H}:force_original_aspect_ratio=decrease:flags=lanczos[fg{k}];"
                f"[bg{k}][fg{k}]overlay=(W-w)/2:(H-h)/2,fps={FPS}[fit{k}]"
            )
            chain = f"[fit{k}]{kb}"
        else:
            raise SystemExit(f"unknown kind {kind}")

        # caption
        cap = s.get("caption")
        if cap:
            cap_file = os.path.join(caption_dir, f"{s['id']}.txt")
            open(cap_file, "w").write(cap)
            color = "0x2ee59d" if cap.upper().startswith("OBSERVED") else ("0xffc14d" if "GENERATED" in cap.upper() else "white")
            chain += (
                f",drawtext=fontfile={FONT}:textfile={cap_file}:fontsize=44:fontcolor={color}"
                f":box=1:boxcolor=0x06080f@0.72:boxborderw=24:x=110:y=h-200"
                f":alpha='clip((t-0.5)/0.6,0,1)'"
            )
        if k == 0:
            chain += ",fade=t=in:st=0:d=0.9"
        if k == n - 1:
            chain += f",fade=t=out:st={dur - 1.3:.2f}:d=1.3"
        chain += f",trim=duration={dur},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[{label}]"
        fc.append(chain)

    # --- crossfade chain ------------------------------------------------------
    prev = "v0"
    acc = scenes[0]["dur"]
    for k in range(1, n):
        trans = TRANSITIONS[(k - 1) % len(TRANSITIONS)]
        offset = round(acc - XF, 3)
        out = f"x{k}" if k < n - 1 else "vout"
        fc.append(f"[{prev}][v{k}]xfade=transition={trans}:duration={XF}:offset={offset}[{out}]")
        prev = out
        acc = offset + scenes[k]["dur"]

    # --- audio ----------------------------------------------------------------
    narr_labels = []
    for k, s in enumerate(scenes):
        i = add_input("-i", s["narr_path"])
        gain = 10 ** ((-17.0 - mean_volume_db(s["narr_path"])) / 20)  # normalize each line to ~-17 dBFS mean
        delay_ms = int((starts[k] + NARR_LEAD) * 1000)
        fc.append(
            f"[{i}:a]aformat=sample_rates=48000:channel_layouts=stereo,volume={gain:.3f},"
            f"adelay={delay_ms}|{delay_ms}[n{k}]"
        )
        narr_labels.append(f"[n{k}]")
    fc.append(f"{''.join(narr_labels)}amix=inputs={n}:duration=longest:normalize=0,apad=whole_dur={total}[narr]")
    fc.append("[narr]asplit[narr_mix][narr_key]")

    mi = add_input("-i", os.path.join(assets, "music.wav"))
    fc.append(
        f"[{mi}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=duration={total},"
        f"afade=t=in:st=0:d=1.2,afade=t=out:st={total - 3.0:.2f}:d=3.0,volume={MUSIC_GAIN}[music]"
    )
    fc.append(
        "[music][narr_key]sidechaincompress=threshold=0.03:ratio=6:attack=40:release=700:level_sc=1.0:mix=1[duck]"
    )
    fc.append("[duck][narr_mix]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.93:level=false[aout]")

    args += [
        "-filter_complex", ";".join(fc),
        "-map", "[vout]", "-map", "[aout]",
        "-t", f"{total}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        out_path,
    ]
    open(os.path.join(assets, "_ffmpeg_cmd.txt"), "w").write(" \\\n  ".join(a if " " not in a else repr(a) for a in args))
    print("rendering…", flush=True)
    subprocess.run(args, check=True)
    print("wrote", out_path, f"({probe_duration(out_path):.1f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
