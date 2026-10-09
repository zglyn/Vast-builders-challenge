"""Sprint tests: inpaint erase, alpha matte, scale/rotate motion orient."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from video_edit import (  # noqa: E402
    build_alpha_matte,
    cv2_available,
    erase_bbox,
    fill_rect,
    ghost_motion,
    make_frame,
    transform_patch,
    crop_patch,
)


class TestInpaintErase(unittest.TestCase):
    def test_erase_removes_bright_blob(self):
        fr = make_frame(80, 60, (50, 55, 60))
        fill_rect(fr, [20, 15, 50, 40], (240, 20, 20))
        # sample center before
        w = fr["w"]
        i = (27 * w + 35) * 3
        self.assertGreater(fr["rgb"][i], 200)
        erase_bbox(fr, [20, 15, 50, 40], inflate=2)
        # after erase, center should be closer to road gray than pure red
        self.assertLess(fr["rgb"][i], 180)


class TestAlphaMatte(unittest.TestCase):
    def test_ellipse_soft_edges(self):
        m = build_alpha_matte(40, 20, feather=0.3)
        self.assertEqual(len(m), 800)
        # center opaque
        self.assertGreater(m[10 * 40 + 20], 0.95)
        # corner transparent
        self.assertLess(m[0], 0.05)


class TestMotionTransform(unittest.TestCase):
    def test_ghost_motion_and_scale_rotate(self):
        path = [
            {"t": 0.0, "cx": 0.0, "cy": 0.0},
            {"t": 1.0, "cx": 100.0, "cy": 0.0},
        ]
        vx, vy, sp = ghost_motion(path, 0.5)
        self.assertAlmostEqual(vx, 100.0, places=1)
        self.assertAlmostEqual(sp, 100.0, places=1)

        fr = make_frame(60, 40, (0, 0, 0))
        fill_rect(fr, [10, 10, 40, 28], (20, 120, 200))
        patch = crop_patch(fr, [10, 10, 40, 28])
        scaled = transform_patch(patch, scale=1.5, angle_deg=0.0)
        self.assertGreater(scaled["w"], patch["w"])
        rotated = transform_patch(patch, scale=1.0, angle_deg=20.0)
        # rotation expands canvas
        self.assertGreaterEqual(rotated["w"] * rotated["h"], patch["w"] * patch["h"])

    def test_cv2_flag_is_bool(self):
        self.assertIsInstance(cv2_available(), bool)


if __name__ == "__main__":
    unittest.main()
