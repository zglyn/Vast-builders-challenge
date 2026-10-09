"""Comic gag overlays + creative what-if parsing (CPU-only, labeled comedy)."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from comic_fx import (  # noqa: E402
    COMIC_DISCLAIMER,
    apply_comic_overlays,
    impact_moment,
    mux_comic_audio,
    pick_gag,
    _write_tone_wav,
)
from edit_spec import build_edit_spec  # noqa: E402
from ensemble import run_ensemble  # noqa: E402
from ghost_engine import Track  # noqa: E402
from video_edit import (  # noqa: E402
    frames_to_mp4,
    get_job,
    make_frame,
    start_generate_job,
)
from whatif_parser import parse_whatif  # noqa: E402


def _toy_analysis() -> dict:
    def toy_track(tid: int, x0: float, y0: float, vx: float = 40.0, n: int = 30) -> Track:
        tr = Track(id=tid, label="car", bbox=[x0 - 20, y0 - 10, x0 + 20, y0 + 10], conf=0.9)
        for i in range(n):
            t = i / 30.0
            cx, cy = x0 + vx * t, y0
            bb = [cx - 20, cy - 10, cx + 20, cy + 10]
            tr.push(i, t, bb, 0.9)
        return tr

    tracks = []
    for tr in (toy_track(1, 100, 200, 50), toy_track(2, 220, 200, 20)):
        tracks.append(
            {
                "id": tr.id,
                "label": tr.label,
                "hits": tr.hits,
                "path": tr.history[::2],
                "full_path": tr.history,
            }
        )
    event = {
        "kind": "close_interaction",
        "t": 0.5,
        "frame": 15,
        "id_a": 1,
        "id_b": 2,
        "label_a": "truck",
        "label_b": "car",
    }
    return {"fps": 30.0, "tracks": tracks, "primary_event": event, "events": [event]}


def _ensemble(scenario: dict):
    analysis = _toy_analysis()
    result = run_ensemble(
        analysis=analysis,
        event=analysis["primary_event"],
        scenario=scenario,
        n_runs=1,
        seed=7,
        llm_chat=None,
        use_agents=True,
    )
    return analysis, result


class TestComicFx(unittest.TestCase):
    def test_pick_gag_swerve(self):
        self.assertEqual(
            pick_gag({"scenario": {"intervention": "delayed_lane_change"}}),
            "swerve",
        )
        self.assertEqual(
            pick_gag({"scenario": {"intervention": "stronger_braking"}}),
            "crash",
        )

    def test_impact_and_overlay(self):
        sc = {
            "targetTrackIds": [1, 2],
            "intervention": "stronger_braking",
            "params": {"brakeIntensity": 0.95, "delaySec": 0.1},
            "horizonSec": 1.5,
            "rationale": "crash gag",
        }
        analysis, ens = _ensemble(sc)
        self.assertTrue(ens.get("ok"))
        spec = build_edit_spec(
            source="test",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        hit = impact_moment(spec)
        self.assertIsNotNone(hit)
        t0 = spec["event_t"]
        frames = [(t0 + dt, make_frame(160, 90, (30, 30, 30))) for dt in (0.0, 0.2, 0.4, 0.6)]
        meta = apply_comic_overlays(frames, spec)
        self.assertEqual(meta["gag"], "crash")
        self.assertIsNotNone(meta["impact_t"])
        self.assertIn("COMEDIC", meta["disclaimer"])
        fr0 = frames[0][1]
        self.assertNotEqual(bytes(fr0["rgb"][0:3]), bytes([30, 30, 30]))

    def test_sfx_wav_and_mux(self):
        with tempfile.TemporaryDirectory() as td:
            wav = os.path.join(td, "crash.wav")
            _write_tone_wav(wav, kind="crash", duration=0.2)
            self.assertTrue(os.path.isfile(wav))
            self.assertGreater(os.path.getsize(wav), 100)
            frames = [
                (0.0, make_frame(64, 36, (10, 10, 10))),
                (0.2, make_frame(64, 36, (10, 10, 10))),
            ]
            silent = os.path.join(td, "silent.mp4")
            out = os.path.join(td, "out.mp4")
            frames_to_mp4(frames, silent, fps=5.0)
            mux_comic_audio(silent, out, impact_t=0.1, duration=0.4)
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 200)

    def test_generate_job_comic_default(self):
        sc = {
            "targetTrackIds": [1],
            "intervention": "delayed_lane_change",
            "params": {"steerIntensity": 0.85, "delaySec": 0.1},
            "horizonSec": 1.5,
            "rationale": "swerve comic",
        }
        analysis, ens = _ensemble(sc)
        spec = build_edit_spec(
            source="test",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        for tr in spec["tracks"]:
            tr["appearance"]["bbox"] = [40, 60, 80, 95]
            tr["appearance"]["cx"] = 60
            tr["appearance"]["cy"] = 77
            t0 = spec["event_t"]
            tr["ghost_path"] = [
                {"t": t0, "frame": 0, "cx": 60, "cy": 77},
                {"t": t0 + 1.0, "frame": 5, "cx": 120, "cy": 70},
            ]
        job = start_generate_job(spec, synthetic=True, comic_mode=True)
        deadline = time.time() + 45
        final = None
        while time.time() < deadline:
            final = get_job(job["id"])
            if final and final.get("status") in ("done", "error"):
                break
            time.sleep(0.15)
        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "done", final.get("error"))
        self.assertTrue(final["result"].get("comic_mode"))
        self.assertIn("comic", final["result"]["mode"])
        self.assertIn("COMEDY", final["report"]["label"])
        self.assertIn("COMEDIC", final["report"]["disclaimer"])
        self.assertEqual(final["result"]["comic"]["gag"], "swerve")


class TestCreativeParse(unittest.TestCase):
    def test_truck_breaks(self):
        analysis = _toy_analysis()
        out = parse_whatif(
            "what if the truck breaks and the cars almost crash?",
            event=analysis["primary_event"],
            known_track_ids={1, 2},
            llm_chat=None,
        )
        self.assertTrue(out["ok"])
        self.assertIn(
            out["scenario"]["intervention"],
            {"earlier_braking", "stronger_braking"},
        )

    def test_disclaimer_constant(self):
        self.assertIn("Not a crash prediction", COMIC_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
