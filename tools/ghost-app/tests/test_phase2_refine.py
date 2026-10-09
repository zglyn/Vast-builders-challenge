"""Phase 2 generative refine tests (Poisson seamless + remote hook)."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from edit_spec import validate_edit_spec  # noqa: E402
from gen_refine import (  # noqa: E402
    remote_inpaint_available,
    render_generative_edit,
    seamless_paste,
)
from video_edit import clone_frame, crop_patch, fill_rect, make_frame  # noqa: E402


def _mini_spec():
    return validate_edit_spec(
        {
            "source": "s3://test/demo.mp4",
            "event_t": 0.5,
            "duration_sec": 0.6,
            "fps": 5,
            "tracks": [
                {
                    "track_id": 1,
                    "label": "car",
                    "appearance": {
                        "t": 0.5,
                        "frame": 0,
                        "bbox": [30, 20, 70, 45],
                        "cx": 50,
                        "cy": 32,
                    },
                    "ghost_path": [
                        {"t": 0.5, "frame": 0, "cx": 50, "cy": 32},
                        {"t": 1.1, "frame": 3, "cx": 110, "cy": 32},
                    ],
                    "observed_boxes": [
                        {"t": 0.5, "frame": 0, "bbox": [30, 20, 70, 45]},
                        {"t": 0.7, "frame": 1, "bbox": [40, 20, 80, 45]},
                        {"t": 0.9, "frame": 2, "bbox": [50, 20, 90, 45]},
                    ],
                }
            ],
        }
    )


class TestPhase2(unittest.TestCase):
    def test_remote_flag_default_off(self):
        os.environ.pop("GHOST_GEN_INPAINT_URL", None)
        self.assertFalse(remote_inpaint_available())

    def test_seamless_paste_runs(self):
        fr = make_frame(200, 100, (60, 65, 70))
        fill_rect(fr, [20, 20, 60, 50], (30, 140, 220))
        patch = crop_patch(fr, [20, 20, 60, 50])
        road = make_frame(200, 100, (60, 65, 70))
        dest, method = seamless_paste(road, patch, 120, 40)
        self.assertIn(method, {"seamless", "matte"})
        self.assertEqual(len(dest), 4)

    def test_generative_render_pipeline(self):
        spec = _mini_spec()
        # build base frames with a moving observed car
        frames = []
        patch_src = make_frame(200, 100, (60, 65, 70))
        fill_rect(patch_src, [30, 20, 70, 45], (25, 120, 200))
        fill_rect(patch_src, [35, 24, 50, 34], (200, 200, 210))
        patch = crop_patch(patch_src, [30, 20, 70, 45])
        for i in range(4):
            t = 0.5 + i * 0.2
            fr = make_frame(200, 100, (60, 65, 70))
            fill_rect(fr, [30 + i * 10, 20, 70 + i * 10, 45], (25, 120, 200))
            frames.append((t, fr))
        edited, stats = render_generative_edit(
            frames,
            spec,
            appearance_patches={1: patch},
            temporal_polish=True,
            stamp=True,
            use_remote_inpaint=False,
        )
        self.assertEqual(len(edited), len(frames))
        self.assertIn("seamless_pastes", stats)
        self.assertTrue(stats["backend"])
        # Generated frames should differ from pure road in later frames
        last = edited[-1][1]
        # some non-road pixels remain (vehicle or banner)
        rgb = last["rgb"]
        colorful = sum(1 for i in range(0, len(rgb), 3) if abs(rgb[i] - 60) > 20)
        self.assertGreater(colorful, 10)


if __name__ == "__main__":
    unittest.main()
