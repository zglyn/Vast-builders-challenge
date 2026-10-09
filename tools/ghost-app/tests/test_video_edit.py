"""Tests for EditSpec + Phase-1 geometric video export."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edit_spec import EditSpecError, build_edit_spec, validate_edit_spec  # noqa: E402
from ensemble import run_ensemble  # noqa: E402
from ghost_engine import Track  # noqa: E402
from video_edit import (  # noqa: E402
    appearance_patches_from_frames,
    blend_patches,
    blit_centered,
    crop_patch,
    erase_bbox,
    export_geometric_synthetic,
    get_job,
    interpolate_ghost,
    load_real_footage,
    make_frame,
    make_split_frames,
    mask_stats,
    read_ppm,
    render_geometric_edit,
    scale_edit_spec,
    start_generate_job,
    synthesize_demo_frames,
    translate_patch,
    write_ppm,
)


def _toy_analysis() -> dict:
    def toy_track(tid: int, x0: float, y0: float, vx: float = 40.0, n: int = 30) -> Track:
        tr = Track(id=tid, label="car", bbox=[x0 - 20, y0 - 10, x0 + 20, y0 + 10], conf=0.9)
        for i in range(n):
            t = i / 30.0
            cx, cy = x0 + vx * t, y0
            bb = [cx - 20, cy - 10, cx + 20, cy + 10]
            tr.push(i, t, bb, 0.9)
        return tr

    t1, t2 = toy_track(1, 100, 200, 50), toy_track(2, 220, 200, 20)
    tracks = []
    for tr in (t1, t2):
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
        "label_a": "car",
        "label_b": "car",
    }
    return {"fps": 30.0, "tracks": tracks, "primary_event": event, "events": [event]}


def _ensemble():
    analysis = _toy_analysis()
    sc = {
        "targetTrackIds": [1, 2],
        "intervention": "earlier_braking",
        "params": {"delaySec": 0.1, "brakeIntensity": 0.8},
        "horizonSec": 1.0,
        "rationale": "test",
    }
    result = run_ensemble(
        analysis=analysis,
        event=analysis["primary_event"],
        scenario=sc,
        n_runs=2,
        seed=3,
        llm_chat=None,
        use_agents=True,
    )
    return analysis, sc, result


class TestEditSpec(unittest.TestCase):
    def test_build_and_validate(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            run_index=0,
            duration_sec=3.0,
            fps=10.0,
        )
        self.assertEqual(spec["label"], "HYPOTHETICAL")
        self.assertEqual(spec["mode"], "geometric_move")
        self.assertEqual(spec["duration_sec"], 3.0)
        self.assertGreaterEqual(len(spec["tracks"]), 1)
        for tr in spec["tracks"]:
            self.assertGreaterEqual(len(tr["ghost_path"]), 2)
            self.assertEqual(len(tr["appearance"]["bbox"]), 4)
        again = validate_edit_spec(spec)
        self.assertEqual(again["source"], spec["source"])

    def test_rejects_empty_ghosts(self):
        with self.assertRaises(EditSpecError):
            validate_edit_spec(
                {
                    "source": "s3://x",
                    "event_t": 0.5,
                    "tracks": [{"track_id": 1, "ghost_path": [], "appearance": {"bbox": [0, 0, 1, 1]}}],
                }
            )

    def test_duration_clamped(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=99.0,
        )
        self.assertLessEqual(spec["duration_sec"], 5.0)


class TestGeometry(unittest.TestCase):
    def test_interpolate_ghost(self):
        path = [
            {"t": 0.0, "cx": 0.0, "cy": 10.0},
            {"t": 1.0, "cx": 100.0, "cy": 10.0},
        ]
        cx, cy = interpolate_ghost(path, 0.5)
        self.assertAlmostEqual(cx, 50.0)
        self.assertAlmostEqual(cy, 10.0)

    def test_erase_and_blit_moves_patch(self):
        fr = make_frame(100, 60, (10, 10, 10))
        # paint a blue vehicle at left
        from video_edit import fill_rect

        fill_rect(fr, [10, 20, 30, 40], (0, 0, 255))
        patch = crop_patch(fr, [10, 20, 30, 40])
        erase_bbox(fr, [10, 20, 30, 40], inflate=0)
        # center of erased region should no longer be pure blue
        w = fr["w"]
        i = (30 * w + 20) * 3
        self.assertNotEqual(tuple(fr["rgb"][i : i + 3]), (0, 0, 255))
        dest = blit_centered(fr, patch, 70, 30)
        self.assertGreater(dest[0], 40)
        # pasted center roughly blue
        cx, cy = 70, 30
        j = (cy * w + cx) * 3
        self.assertEqual(tuple(fr["rgb"][j : j + 3]), (0, 0, 255))


class TestExportPipeline(unittest.TestCase):
    def test_render_moves_along_ghost(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        # Shrink to toy canvas for synthetic
        for tr in spec["tracks"]:
            tid = tr["track_id"]
            # map appearance into 320x180 space roughly
            tr["appearance"]["bbox"] = [40 + tid * 30, 80, 70 + tid * 30, 110]
            tr["appearance"]["cx"] = (tr["appearance"]["bbox"][0] + tr["appearance"]["bbox"][2]) / 2
            tr["appearance"]["cy"] = (tr["appearance"]["bbox"][1] + tr["appearance"]["bbox"][3]) / 2
            # ghost moves right
            t0 = spec["event_t"]
            tr["ghost_path"] = [
                {"t": t0, "frame": 0, "cx": tr["appearance"]["cx"], "cy": tr["appearance"]["cy"]},
                {"t": t0 + 1.0, "frame": 5, "cx": tr["appearance"]["cx"] + 80, "cy": tr["appearance"]["cy"]},
            ]
            tr["observed_boxes"] = []

        base, patches = synthesize_demo_frames(spec, width=320, height=180)
        # fill observed boxes for erase
        for tr in spec["tracks"]:
            b = tr["appearance"]["bbox"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            tr["observed_boxes"] = [
                {
                    "t": t,
                    "frame": 0,
                    "bbox": [
                        tr["appearance"]["cx"] + (t - spec["event_t"]) * 30 - bw / 2,
                        tr["appearance"]["cy"] - bh / 2,
                        tr["appearance"]["cx"] + (t - spec["event_t"]) * 30 + bw / 2,
                        tr["appearance"]["cy"] + bh / 2,
                    ],
                }
                for t, _ in base
            ]
        edited = render_geometric_edit(base, spec, appearance_patches=patches)
        self.assertEqual(len(edited), len(base))
        # Last frame: patch should be near ghost end (further right than start)
        _t0, first = edited[0]
        _t1, last = edited[-1]
        tid = spec["tracks"][0]["track_id"]
        patch = patches[tid]
        # sample: count blue-ish / non-road pixels in right half vs left half on last frame
        w, h, rgb = last["w"], last["h"], last["rgb"]
        right = left = 0
        for y in range(h):
            for x in range(w):
                i = (y * w + x) * 3
                # non-road if not near background gray
                if abs(rgb[i] - 48) > 15 or abs(rgb[i + 1] - 52) > 15:
                    if x > w // 2:
                        right += 1
                    else:
                        left += 1
        self.assertGreater(right, 0)
        # ghost moves right — expect more non-road mass on the right by the end
        self.assertGreaterEqual(right, left // 2)

    @unittest.skipUnless(
        subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        "ffmpeg not installed",
    )
    def test_ffmpeg_export_creates_mp4(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        for tr in spec["tracks"]:
            tr["appearance"]["bbox"] = [50, 70, 90, 100]
            tr["appearance"]["cx"] = 70
            tr["appearance"]["cy"] = 85
            t0 = spec["event_t"]
            tr["ghost_path"] = [
                {"t": t0, "frame": 0, "cx": 70, "cy": 85},
                {"t": t0 + 1.0, "frame": 5, "cx": 140, "cy": 85},
            ]
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "out.mp4")
            result = export_geometric_synthetic(spec, out)
            self.assertTrue(result["ok"])
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 500)
            # ffprobe duration roughly 1s
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    out,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                dur = float(probe.stdout.strip())
                self.assertGreater(dur, 0.4)
                self.assertLess(dur, 2.5)

    @unittest.skipUnless(
        subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        "ffmpeg not installed",
    )
    def test_async_job_completes(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
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
                {"t": t0 + 1.0, "frame": 5, "cx": 120, "cy": 77},
            ]
        job = start_generate_job(spec, synthetic=True, temporal_polish=True, split_screen=True)
        job_id = job["id"]
        deadline = time.time() + 30
        final = None
        while time.time() < deadline:
            final = get_job(job_id)
            if final and final.get("status") in ("done", "error"):
                break
            time.sleep(0.2)
        self.assertIsNotNone(final)
        self.assertEqual(final["status"], "done", final.get("error"))
        self.assertTrue(os.path.isfile(final["path"]))
        self.assertIsNotNone(final.get("report"))
        self.assertIn("temporal", final["result"]["mode"])
        self.assertIn("split", final["result"]["mode"])
        self.assertIn("mask_stats", final["report"])


class TestPhase3Temporal(unittest.TestCase):
    def test_translate_and_blend(self):
        fr = make_frame(40, 40, (0, 0, 0))
        from video_edit import fill_rect

        fill_rect(fr, [5, 5, 15, 15], (255, 0, 0))
        patch = crop_patch(fr, [5, 5, 15, 15])
        moved = translate_patch(patch, 3, 0)
        self.assertEqual(moved["w"], patch["w"])
        # Blended should be between
        mixed = blend_patches(patch, moved, 0.5)
        self.assertEqual(mixed["w"], patch["w"])

    def test_temporal_render_runs(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        for tr in spec["tracks"]:
            tr["appearance"]["bbox"] = [40 + tr["track_id"] * 20, 70, 70 + tr["track_id"] * 20, 100]
            tr["appearance"]["cx"] = (tr["appearance"]["bbox"][0] + tr["appearance"]["bbox"][2]) / 2
            tr["appearance"]["cy"] = 85
            t0 = spec["event_t"]
            tr["ghost_path"] = [
                {"t": t0, "frame": 0, "cx": tr["appearance"]["cx"], "cy": 85},
                {"t": t0 + 1.0, "frame": 5, "cx": tr["appearance"]["cx"] + 60, "cy": 85},
            ]
            tr["observed_boxes"] = []
        base, patches = synthesize_demo_frames(spec, width=320, height=180)
        for tr in spec["tracks"]:
            b = tr["appearance"]["bbox"]
            bw, bh = b[2] - b[0], b[3] - b[1]
            tr["observed_boxes"] = [
                {
                    "t": t,
                    "frame": 0,
                    "bbox": [
                        tr["appearance"]["cx"] + (t - spec["event_t"]) * 20 - bw / 2,
                        85 - bh / 2,
                        tr["appearance"]["cx"] + (t - spec["event_t"]) * 20 + bw / 2,
                        85 + bh / 2,
                    ],
                }
                for t, _ in base
            ]
        plain = render_geometric_edit(base, spec, appearance_patches=patches, temporal_polish=False)
        polished = render_geometric_edit(base, spec, appearance_patches=patches, temporal_polish=True)
        self.assertEqual(len(plain), len(polished))
        # Frames should differ somewhere when polish blends warped history
        differ = False
        for (_t1, a), (_t2, b) in zip(plain[1:], polished[1:]):
            if a["rgb"] != b["rgb"]:
                differ = True
                break
        self.assertTrue(differ)


class TestPhase4Product(unittest.TestCase):
    def test_split_frames_double_width(self):
        left = [(0.0, make_frame(100, 50, (0, 80, 0))), (0.1, make_frame(100, 50, (0, 90, 0)))]
        right = [(0.0, make_frame(100, 50, (80, 0, 0))), (0.1, make_frame(100, 50, (90, 0, 0)))]
        split = make_split_frames(left, right)
        self.assertEqual(len(split), 2)
        self.assertEqual(split[0][1]["w"], 200)
        self.assertEqual(split[0][1]["h"], 50)

    def test_mask_stats(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
        )
        stats = mask_stats(spec)
        self.assertGreaterEqual(len(stats), 1)
        self.assertIn("appearance_area_px", stats[0])

    @unittest.skipUnless(
        subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        "ffmpeg not installed",
    )
    def test_export_split_and_temporal_mp4(self):
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "split.mp4")
            result = export_geometric_synthetic(
                spec, out, temporal_polish=True, split_screen=True
            )
            self.assertTrue(result["ok"])
            self.assertTrue(result["temporal_polish"])
            self.assertTrue(result["split_screen"])
            self.assertIn("split", result["mode"])
            self.assertTrue(os.path.isfile(out))
            self.assertGreater(os.path.getsize(out), 500)
            self.assertTrue(result["mask_stats"])

    @unittest.skipUnless(
        subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        "ffmpeg not installed",
    )
    def test_fallback_path_still_exports(self):
        """If polish path is forced off, Phase-1 fallback still produces a file."""
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "plain.mp4")
            result = export_geometric_synthetic(
                spec, out, temporal_polish=False, split_screen=False
            )
            self.assertTrue(result["ok"])
            self.assertFalse(result["temporal_polish"])
            self.assertTrue(os.path.isfile(out))


class TestRealFootage(unittest.TestCase):
    def test_ppm_roundtrip_and_scale_spec(self):
        fr = make_frame(32, 24, (10, 20, 30))
        from video_edit import fill_rect

        fill_rect(fr, [4, 4, 12, 12], (200, 100, 50))
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "t.ppm")
            write_ppm(path, fr)
            back = read_ppm(path)
            self.assertEqual(back["w"], 32)
            self.assertEqual(back["h"], 24)
            self.assertEqual(bytes(back["rgb"]), bytes(fr["rgb"]))
        analysis, sc, ens = _ensemble()
        spec = build_edit_spec(
            source="s3://test/demo.mp4",
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            ensemble_result=ens,
        )
        scaled = scale_edit_spec(spec, 0.5, 0.5)
        self.assertAlmostEqual(
            scaled["tracks"][0]["appearance"]["cx"],
            spec["tracks"][0]["appearance"]["cx"] * 0.5,
        )

    def test_crop_real_pixels_not_flat_color(self):
        """Appearance patch from a varied frame should keep multiple colors (not a solid debug box)."""
        fr = make_frame(80, 60, (40, 40, 40))
        from video_edit import fill_rect

        # fake "vehicle" with two tones
        fill_rect(fr, [20, 20, 50, 40], (30, 90, 180))
        fill_rect(fr, [22, 22, 35, 30], (220, 220, 230))  # windshield-ish
        from video_edit import clone_frame

        frames = [(0.5, fr), (0.6, clone_frame(fr))]
        spec = {
            "source": "s3://x",
            "event_t": 0.5,
            "duration_sec": 1.0,
            "fps": 5,
            "tracks": [
                {
                    "track_id": 1,
                    "label": "car",
                    "appearance": {"t": 0.5, "frame": 0, "bbox": [20, 20, 50, 40], "cx": 35, "cy": 30},
                    "ghost_path": [
                        {"t": 0.5, "frame": 0, "cx": 35, "cy": 30},
                        {"t": 1.0, "frame": 5, "cx": 55, "cy": 30},
                    ],
                    "observed_boxes": [],
                }
            ],
        }
        from edit_spec import validate_edit_spec

        spec = validate_edit_spec(spec)
        patches = appearance_patches_from_frames(frames, spec)
        rgb = patches[1]["rgb"]
        colors = {(rgb[i], rgb[i + 1], rgb[i + 2]) for i in range(0, len(rgb), 3)}
        self.assertGreaterEqual(len(colors), 2)

    @unittest.skipUnless(
        os.path.isfile("/tmp/ghost-demo.mp4")
        and subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
        "needs /tmp/ghost-demo.mp4 + ffmpeg",
    )
    def test_load_real_i24_footage(self):
        analysis, sc, ens = _ensemble()
        # Use real demo primary timing by overlaying real-ish coords from demo ranges
        spec = build_edit_spec(
            source="s3://team-42-vss-chunks-segments/segments/demo.mp4",
            analysis=analysis,
            event={"kind": "close_interaction", "t": 3.5, "frame": 105, "id_a": 1, "id_b": 2},
            scenario=sc,
            ensemble_result=ens,
            duration_sec=1.0,
            fps=5.0,
        )
        # Place appearance in real 1920x1080-ish space so scaling works
        for i, tr in enumerate(spec["tracks"]):
            x = 400 + i * 200
            tr["appearance"] = {"t": 3.5, "frame": 105, "bbox": [x, 400, x + 120, 500], "cx": x + 60, "cy": 450}
            tr["ghost_path"] = [
                {"t": 3.5, "frame": 105, "cx": x + 60, "cy": 450},
                {"t": 4.5, "frame": 135, "cx": x + 200, "cy": 450},
            ]
            tr["observed_boxes"] = [
                {"t": 3.5 + k * 0.2, "frame": 105 + k, "bbox": [x + k * 20, 400, x + 120 + k * 20, 500]}
                for k in range(5)
            ]
        base, patches, scaled = load_real_footage(spec, "/tmp/ghost-demo.mp4", max_width=640)
        self.assertGreaterEqual(len(base), 3)
        self.assertIn(spec["tracks"][0]["track_id"], patches)
        # patch should not be tiny
        self.assertGreater(patches[spec["tracks"][0]["track_id"]]["w"], 10)
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "real.mp4")
            job = start_generate_job(
                spec,
                synthetic=False,
                video_path="/tmp/ghost-demo.mp4",
                temporal_polish=True,
                split_screen=True,
            )
            deadline = time.time() + 60
            final = None
            while time.time() < deadline:
                final = get_job(job["id"])
                if final and final.get("status") in ("done", "error"):
                    break
                time.sleep(0.3)
            self.assertEqual(final["status"], "done", final.get("error"))
            self.assertEqual(final["result"].get("footage"), "real_footage")
            self.assertIn("real_footage", final["result"]["mode"])
            self.assertTrue(os.path.isfile(final["path"]))
            self.assertGreater(os.path.getsize(final["path"]), 5000)


if __name__ == "__main__":
    unittest.main()
