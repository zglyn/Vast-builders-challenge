"""Futures, occlusion, and redesign stay inside the image and stay deterministic."""

from __future__ import annotations

import unittest

from futures import occlusion_zones, redesign_notes, simulate_futures


def _obj(cx, cy, w=90, h=70, label="car", tid=1, vx_norm=0.0):
    box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    traj = []
    for i in range(7):
        traj.append([round((cx / 1920) + vx_norm * 0.12 * i, 4), round(cy / 1080, 4)])
    return {
        "id": tid,
        "label": label,
        "display": label.upper(),
        "bbox": box,
        "trajectory": traj,
        "uncertainty": 0.02,
        "risk": 60,
        "conflict": True,
        "factors": {"closing_motion": 0.8, "trajectory_overlap": 0.7},
    }


def _frame():
    return {
        "t": 1.0,
        "corridor_mode": "approximate",
        "corridor": [[0.30, 0.97], [0.70, 0.97], [0.545, 0.52], [0.455, 0.52]],
        "objects": [],
    }


class FutureTests(unittest.TestCase):
    def test_delay_keeps_early_paths_together(self):
        obj = _obj(1100, 780)
        pack = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.45, brake=0.35, horizon_s=1.2)
        paths = {s["id"]: s["path"] for s in pack["scenarios"]}
        self.assertEqual(paths["continue"][0], paths["brake"][0])
        self.assertEqual(paths["continue"][1], paths["evade"][1])
        self.assertNotEqual(paths["continue"][-1], paths["brake"][-1])
        self.assertNotEqual(paths["continue"][-1], paths["evade"][-1])

    def test_brake_stops_short_of_continue(self):
        obj = _obj(960, 810)
        pack = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.0, brake=0.35, horizon_s=1.2)
        by = {s["id"]: s for s in pack["scenarios"]}
        self.assertGreater(by["brake"]["path"][-1][1], by["continue"]["path"][-1][1])
        self.assertGreater(by["brake"]["min_distance"], by["continue"]["min_distance"])

    def test_evade_moves_away_from_the_object(self):
        obj = _obj(1300, 800)
        pack = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.0, brake=0.4, horizon_s=1.2)
        self.assertEqual(pack["evade_direction"], "left")
        by = {s["id"]: s for s in pack["scenarios"]}
        self.assertLess(by["evade"]["path"][-1][0], by["continue"]["path"][-1][0])

    def test_horizon_changes_sampling(self):
        obj = _obj(960, 800, vx_norm=0.05)
        short = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.0, horizon_s=0.5)
        long = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.0, horizon_s=1.6)
        self.assertNotEqual(short["scenarios"][0]["object_path"][-1], long["scenarios"][0]["object_path"][-1])

    def test_deterministic(self):
        obj = _obj(1000, 790)
        a = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.2, brake=0.5, horizon_s=1.0)
        b = simulate_futures(_frame(), obj, 1920, 1080, delay_s=0.2, brake=0.5, horizon_s=1.0)
        self.assertEqual(a, b)

    def test_large_low_vehicle_occludes_without_inventing_anyone(self):
        truck = _obj(400, 820, w=280, h=220, label="truck", tid=3)
        frame = _frame()
        frame["objects"] = [truck]
        zones = occlusion_zones(frame, 1920, 1080)
        self.assertTrue(zones["supported"])
        self.assertIn("No hidden", zones["zones"][0]["note"])
        self.assertNotIn("pedestrian detected", zones["zones"][0]["note"].lower())

    def test_distant_box_is_not_a_blind_spot(self):
        car = _obj(960, 280, w=70, h=40, label="car")
        frame = _frame()
        frame["objects"] = [car]
        zones = occlusion_zones(frame, 1920, 1080)
        self.assertFalse(zones["supported"])

    def test_redesign_does_not_claim_a_percent_benefit(self):
        person = _obj(960, 800, label="person", tid=7)
        person["display"] = "PEDESTRIAN"
        frame = _frame()
        frame["objects"] = [person]
        zones = occlusion_zones(frame, 1920, 1080)
        notes = redesign_notes(frame, person, zones)
        blob = str(notes)
        self.assertNotIn("% reduction", blob)
        self.assertIn("not a measured", notes["label"])
        self.assertTrue(any(i["id"] == "daylighting" for i in notes["ideas"]))


if __name__ == "__main__":
    unittest.main()
