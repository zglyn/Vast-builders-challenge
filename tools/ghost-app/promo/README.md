# GHOST promo video

85 s pitch video: narration (problem statement, intro, walkthrough), upbeat synth bed
ducked under the voice, real I-24 footage, real what-if exports and UI shots.

| File | Role |
|------|------|
| `script.json` | Scene list: narration lines, captions, min durations, TTS voice |
| `cards.html` | Title cards (render with headless Chrome, `?card=<id>`) |
| `music.py` | Pure-Python synth-pop bed (124 BPM, I–V–vi–IV) → WAV |
| `tts.py` | One MP3 per scene via `edge-tts` (run where pip is available, e.g. the app pod) |
| `build_promo.py` | ffmpeg assembly: Ken Burns, crossfades, captions, side-chain ducking |

## Rebuild

```bash
A=~/ghost-promo/assets; mkdir -p $A/tts
python3 music.py $A/music.wav 92
for c in problem intro ask honest outro; do
  google-chrome --headless=new --no-sandbox --hide-scrollbars --window-size=1920,1080 \
    --virtual-time-budget=6000 --screenshot=$A/card_$c.png "file://$PWD/cards.html?card=$c"
done
# With the GHOST app port-forwarded on :18080:
#   assets/demo_clip.mp4      GET /api/video?source=<demo_source>
#   assets/export_quality.mp4 POST /api/whatif/generate (quality_mode, comic off)
#   assets/export_split.mp4   same with split_screen=true
#   assets/shot_lab.png / shot_creative.png  headless Chrome screenshots of / and /creative
#   assets/tts/<scene>.mp3    python tts.py script.json tts/  (in the pod), then kubectl cp
python3 build_promo.py script.json $A ~/ghost-promo/out/GHOST_promo.mp4
```

Rendered output: [`docs/media/GHOST_promo.mp4`](../../../docs/media/GHOST_promo.mp4) (≈31 MB).
