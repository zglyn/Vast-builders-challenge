"""Deterministic checks for GHOST NYC tracking and the heuristic risk model."""

from __future__ import annotations

import unittest

from risk_engine import RISK_WEIGHTS, analyze_detections, estimate_foe


def _clip(frames: list[dict], fps: float = 30.0) -> dict:
    return analyze_detections({"fps": fps, "frames": frames, "source": "synthetic"})


def _frame(i: int, dets: list[dict], fps: float = 30.0) -> dict:
    return {
        "frame_index": i,
        "time_sec": i / fps,
        "shape": [1080, 1920],
        "detections": dets,
    }


def _box(cx: float, cy: float, w: float, h: float, label: str = "car", conf: float = 0.86) -> dict:
    return {
        "label": label,
        "confidence": conf,
        "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
    }


def _scores(result: dict, track_id: int = 1) -> list[float]:
    vals = []
    for fr in result["frames"]:
        for obj in fr["objects"]:
            if obj["id"] == track_id and obj["risk"] is not None:
                vals.append(obj["risk"])
    return vals


class RiskModelTests(unittest.TestCase):
    def test_weights_are_normalized(self):
        self.assertAlmostEqual(sum(RISK_WEIGHTS.values()), 1.0)

    def test_deterministic(self):
        frames = [
            _frame(i, [_box(960, 520 + i * 10, 60 + i * 4, 40 + i * 3)])
            for i in range(18)
        ]
        a = _clip(frames)
        b = _clip(frames)
        self.assertEqual(a["frames"], b["frames"])

    def test_approaching_increases(self):
        frames = [
            _frame(i, [_box(960, 500 + i * 14, 48 + i * 7, 32 + i * 5)])
            for i in range(20)
        ]
        scores = _scores(_clip(frames))
        self.assertGreaterEqual(len(scores), 8)
        self.assertGreater(scores[-1], scores[0] + 4)
        self.assertGreater(scores[-1], 50)

    def test_receding_decreases(self):
        frames = [
            _frame(i, [_box(960, 900 - i * 12, 220 - i * 6, 150 - i * 4)])
            for i in range(16)
        ]
        scores = _scores(_clip(frames))
        self.assertGreaterEqual(len(scores), 6)
        self.assertLess(scores[-1], scores[0])

    def test_crossing_rises_above_start(self):
        frames = [
            _frame(i, [_box(40 + i * 22, 840, 70, 56, label="person")])
            for i in range(40)
        ]
        scores = _scores(_clip(frames))
        self.assertGreaterEqual(len(scores), 8)
        self.assertGreater(max(scores), scores[0] + 5)

    def test_stationary_roadside_stays_low(self):
        frames = [_frame(i, [_box(1720, 820, 200, 160)]) for i in range(16)]
        scores = _scores(_clip(frames))
        self.assertTrue(scores)
        self.assertLess(max(scores), 30)

    def test_large_distant_box_is_not_imminent(self):
        frames = [_frame(i, [_box(960, 180, 420, 240)]) for i in range(16)]
        scores = _scores(_clip(frames))
        self.assertTrue(scores)
        self.assertLess(max(scores), 25)

    def test_short_track_is_analyzing(self):
        frames = [_frame(i, [_box(960, 700, 80, 60)]) for i in range(2)]
        result = _clip(frames)
        objs = [o for fr in result["frames"] for o in fr["objects"]]
        self.assertTrue(objs)
        self.assertTrue(all(o["status"] == "ANALYZING" and o["risk"] is None for o in objs))

    def test_missing_detections_keep_id(self):
        frames = []
        for i in range(18):
            dets = []
            if i not in (8, 9, 10, 11):
                dets.append(_box(700 + i * 8, 760, 100, 70))
            frames.append(_frame(i, dets))
        result = _clip(frames)
        ids = {o["id"] for fr in result["frames"] for o in fr["objects"] if o["label"] == "car"}
        self.assertEqual(ids, {1})

    def test_long_gap_mints_a_new_id(self):
        frames = []
        for i in range(28):
            dets = []
            if i < 8 or i >= 24:
                dets.append(_box(960, 780, 100, 70))
            frames.append(_frame(i, dets))
        result = _clip(frames)
        ids = sorted({o["id"] for fr in result["frames"] for o in fr["objects"]})
        self.assertGreaterEqual(len(ids), 2)

    def test_opposing_traffic_outranks_roadside(self):
        frames = []
        for i in range(18):
            frames.append(
                _frame(
                    i,
                    [
                        _box(960, 480 + i * 18, 50 + i * 8, 36 + i * 6, label="bus", conf=0.9),
                        _box(1760, 860, 180, 140, label="car", conf=0.9),
                    ],
                )
            )
        result = _clip(frames)
        last = result["frames"][-1]["objects"]
        bus = next(o for o in last if o["label"] == "bus" and o["risk"] is not None)
        side = next(o for o in last if o["label"] == "car" and o["risk"] is not None)
        self.assertGreater(bus["risk"], side["risk"] + 15)

    def test_focus_of_expansion(self):
        true = (0.42, 0.40)
        samples = []
        for ang in (0.4, 1.1, 2.0, 2.8, 4.0, 5.2):
            px = true[0] + 0.12 * math_cos(ang)
            py = true[1] + 0.10 * math_sin(ang)
            samples.append((px, py, 0.12 * math_cos(ang), 0.10 * math_sin(ang)))
        foe = estimate_foe(samples)
        self.assertIsNotNone(foe)
        self.assertLess(abs(foe[0] - true[0]), 0.08)
        self.assertLess(abs(foe[1] - true[1]), 0.08)

    def test_camera_motion_labels_estimated_corridor(self):
        frames = []
        origins = []
        foe = (0.48 * 1920, 0.42 * 1080)
        for k, ang in enumerate((0.3, 1.2, 2.2, 3.4, 4.4, 5.5)):
            origins.append((foe[0] + 70 * math_cos(ang), foe[1] + 50 * math_sin(ang), ang))
        for i in range(16):
            dets = []
            for ox, oy, ang in origins:
                rad = 30 + i * 7
                cx = ox + rad * math_cos(ang)
                cy = oy + rad * math_sin(ang)
                dets.append(_box(cx, cy, 36, 28, conf=0.8))
            frames.append(_frame(i, dets))
        result = _clip(frames)
        modes = {fr["corridor_mode"] for fr in result["frames"] if any(o["hits"] >= 4 for o in fr["objects"])}
        self.assertIn("estimated", modes)


def math_cos(v: float) -> float:
    import math

    return math.cos(v)


def math_sin(v: float) -> float:
    import math

    return math.sin(v)


if __name__ == "__main__":
    unittest.main()
