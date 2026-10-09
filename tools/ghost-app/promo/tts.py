"""Generate one narration MP3 per scene with edge-tts.

Usage: python tts.py script.json OUT_DIR
"""

import asyncio
import json
import os
import sys

import edge_tts


async def main(script_path: str, out_dir: str) -> None:
    cfg = json.load(open(script_path))
    os.makedirs(out_dir, exist_ok=True)
    for s in cfg["scenes"]:
        com = edge_tts.Communicate(s["narration"], cfg["voice"], rate=cfg["rate"], pitch=cfg["pitch"])
        await com.save(os.path.join(out_dir, f"{s['id']}.mp3"))
        print("ok", s["id"], flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
