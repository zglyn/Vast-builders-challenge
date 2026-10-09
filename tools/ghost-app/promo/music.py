"""Upbeat synth-pop bed for the GHOST promo. Pure Python (no numpy).

Usage: python3 music.py OUT.wav SECONDS
"""

from __future__ import annotations

import math
import random
import sys
import wave
from array import array

SR = 32000
BPM = 124
BEAT = 60.0 / BPM
BAR = BEAT * 4

# I–V–vi–IV in C (bright, familiar, "ad" feel). MIDI notes.
CHORDS = [
    (60, 64, 67),  # C
    (55, 59, 62),  # G
    (57, 60, 64),  # Am
    (53, 57, 60),  # F
]
BASS = [48, 43, 45, 41]


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


class Mix:
    def __init__(self, seconds: float):
        self.n = int(seconds * SR)
        self.buf = array("f", bytes(4 * self.n))

    def add(self, start: float, samples: array, gain: float = 1.0) -> None:
        i0 = int(start * SR)
        buf = self.buf
        n = self.n
        for k, v in enumerate(samples):
            i = i0 + k
            if i >= n:
                break
            if i >= 0:
                buf[i] += v * gain


def env_pluck(n: int, decay: float) -> list[float]:
    return [math.exp(-k / (SR * decay)) for k in range(n)]


def tone(freq: float, dur: float, kind: str, decay: float | None = None, attack: float = 0.0) -> array:
    n = int(dur * SR)
    out = array("f", bytes(4 * n))
    w = 2 * math.pi * freq / SR
    env = env_pluck(n, decay) if decay else [1.0] * n
    atk = int(attack * SR)
    for k in range(n):
        ph = w * k
        if kind == "square":  # soft chiptune square (first 3 odd harmonics)
            v = math.sin(ph) + math.sin(3 * ph) / 3 + math.sin(5 * ph) / 5
        elif kind == "saw":
            v = math.sin(ph) + math.sin(2 * ph) / 2 + math.sin(3 * ph) / 3 + math.sin(4 * ph) / 4
        else:  # sine-ish pad with a touch of 2nd harmonic
            v = math.sin(ph) + 0.35 * math.sin(2 * ph)
        e = env[k]
        if atk and k < atk:
            e *= k / atk
        # release tail to avoid clicks
        if k > n - 400:
            e *= (n - k) / 400
        out[k] = v * e
    return out


def kick(dur: float = 0.25) -> array:
    n = int(dur * SR)
    out = array("f", bytes(4 * n))
    ph = 0.0
    for k in range(n):
        t = k / SR
        f = 48 + 140 * math.exp(-t * 28)
        ph += 2 * math.pi * f / SR
        out[k] = math.sin(ph) * math.exp(-t * 11)
    return out


def noise_hit(dur: float, decay: float, bright: bool, seed: int) -> array:
    rnd = random.Random(seed)
    n = int(dur * SR)
    out = array("f", bytes(4 * n))
    prev = 0.0
    for k in range(n):
        x = rnd.uniform(-1, 1)
        v = (x - prev) if bright else x  # crude high-pass for hats
        prev = x
        out[k] = v * math.exp(-k / (SR * decay))
    return out


def build(seconds: float) -> Mix:
    mix = Mix(seconds + 1.0)
    bars = int(seconds / BAR) + 1
    hat = noise_hit(0.08, 0.025, True, 1)
    hat_open = noise_hit(0.18, 0.07, True, 2)
    snare_noise = noise_hit(0.22, 0.06, False, 3)
    kck = kick()

    for b in range(bars):
        t0 = b * BAR
        chord = CHORDS[b % 4]
        bass = BASS[b % 4]
        intro = b < 2            # arps + pad only
        build_up = 2 <= b < 4    # add bass + hats
        full = b >= 4

        # Pad: whole-bar chord, slow attack
        for m in chord:
            mix.add(t0, tone(hz(m), BAR * 0.98, "pad", attack=0.25), 0.075)

        # Arpeggio: 16ths over chord tones + octave, plucky square
        arp = [chord[0], chord[1], chord[2], chord[1] + 12, chord[2], chord[0] + 12, chord[1], chord[2] + 12]
        for s in range(16):
            m = arp[s % 8] + (12 if full and s % 8 in (3, 5) else 0)
            g = 0.09 if (s % 4 == 0) else 0.065
            mix.add(t0 + s * BEAT / 4, tone(hz(m), 0.22, "square", decay=0.09), g)

        if intro:
            continue

        # Bass: root on 1, 2.5, 3, 4 (bouncy)
        for pos in (0.0, 1.5, 2.0, 3.0, 3.5):
            mix.add(t0 + pos * BEAT, tone(hz(bass), 0.32, "saw", decay=0.14), 0.16)

        # Hats: 8ths, open hat on the "and" of 4
        for s in range(8):
            t = t0 + s * BEAT / 2
            mix.add(t, hat_open if (s == 7 and full) else hat, 0.11 if s % 2 else 0.14)

        if build_up and b == 3:
            # snare roll into the drop
            for s in range(8):
                mix.add(t0 + 2 * BEAT + s * BEAT / 4, snare_noise, 0.10 + 0.03 * s)
            continue

        if full:
            # Four-on-the-floor kick, snare on 2 & 4
            for beat in range(4):
                mix.add(t0 + beat * BEAT, kck, 0.55)
            for beat in (1, 3):
                mix.add(t0 + beat * BEAT, snare_noise, 0.26)
                mix.add(t0 + beat * BEAT, tone(190, 0.12, "pad", decay=0.04), 0.12)

    return mix


def write_wav(path: str, mix: Mix, seconds: float) -> None:
    n = int(seconds * SR)
    peak = max(1e-6, max(abs(v) for v in mix.buf[:n]))
    scale = 0.89 / peak
    fade_in = int(0.8 * SR)
    fade_out = int(2.5 * SR)
    pcm = array("h")
    for i in range(n):
        v = mix.buf[i] * scale
        if i < fade_in:
            v *= i / fade_in
        if i > n - fade_out:
            v *= (n - i) / fade_out
        pcm.append(int(max(-1.0, min(1.0, v)) * 32767))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    out = sys.argv[1]
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 80.0
    write_wav(out, build(secs), secs)
    print(f"wrote {out} ({secs:.1f}s @ {BPM} BPM)")
