"""Unit tests for GHOST what-if / multi-agent extension."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ensemble import cache_key, run_ensemble  # noqa: E402
from ghost_engine import (  # noqa: E402
    Track,
    assess_path_interaction,
    decision_to_trajectory,
    intervention_to_trajectory,
)
from persona_config import PERSONA_NAMES, mid_persona_params, sample_persona_params  # noqa: E402
from scenario_schema import (  # noqa: E402
    ScenarioValidationError,
    validate_agent_decision,
    validate_scenario,
)
from whatif_parser import parse_whatif  # noqa: E402
import random  # noqa: E402


def _toy_track(tid: int, x0: float, y0: float, vx: float = 40.0, n: int = 30) -> Track:
    tr = Track(id=tid, label="car", bbox=[x0 - 20, y0 - 10, x0 + 20, y0 + 10], conf=0.9)
    for i in range(n):
        t = i / 30.0
        cx, cy = x0 + vx * t, y0
        bb = [cx - 20, cy - 10, cx + 20, cy + 10]
        tr.push(i, t, bb, 0.9)
    return tr


def _toy_analysis() -> dict:
    t1 = _toy_track(1, 100, 200, 50)
    t2 = _toy_track(2, 220, 200, 20)
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


class TestScenarioSchema(unittest.TestCase):
    def test_valid_scenario(self):
        sc = validate_scenario(
            {
                "targetTrackIds": [1, 2],
                "intervention": "earlier_braking",
                "params": {"delaySec": 0.1, "brakeIntensity": 0.8},
                "horizonSec": 1.5,
                "rationale": "brake earlier",
            },
            known_track_ids={1, 2, 3},
        )
        self.assertEqual(sc["intervention"], "earlier_braking")
        self.assertEqual(sc["label"], "HYPOTHETICAL")

    def test_unsupported_intervention(self):
        with self.assertRaises(ScenarioValidationError) as ctx:
            validate_scenario({"intervention": "fly_away", "targetTrackIds": [1]})
        self.assertTrue(ctx.exception.unsupported)

    def test_unknown_track(self):
        with self.assertRaises(ScenarioValidationError):
            validate_scenario(
                {"intervention": "maintain_velocity", "targetTrackIds": [99]},
                known_track_ids={1, 2},
            )

    def test_agent_decision_clamps_and_filters_ids(self):
        d = validate_agent_decision(
            {
                "action": "brake",
                "intensity": 2.5,
                "reactionDelaySec": 9,
                "reasoning": "saw #1 and #99",
                "perceivedHazardIds": [1, 99],
            },
            known_ids={1, 2},
        )
        self.assertEqual(d["intensity"], 1.0)
        self.assertEqual(d["reactionDelaySec"], 2.0)
        self.assertEqual(d["perceivedHazardIds"], [1])


class TestPersona(unittest.TestCase):
    def test_mid_and_sample(self):
        for name in PERSONA_NAMES:
            mid = mid_persona_params(name)
            self.assertEqual(mid["persona"], name)
            rng = random.Random(0)
            s = sample_persona_params(name, rng)
            self.assertGreaterEqual(s["braking_intensity"], 0.0)
            self.assertLessEqual(s["braking_intensity"], 1.0)


class TestDecisionTrajectory(unittest.TestCase):
    def test_brake_slows_vs_maintain(self):
        tr = _toy_track(1, 100, 200, 60)
        maintain = decision_to_trajectory(
            tr, 15, {"action": "maintain", "intensity": 0, "reactionDelaySec": 0}, horizon_sec=1.0
        )
        brake = decision_to_trajectory(
            tr, 15, {"action": "brake", "intensity": 0.9, "reactionDelaySec": 0.0}, horizon_sec=1.0
        )
        self.assertIsNotNone(maintain)
        self.assertIsNotNone(brake)
        # braked tip should travel less in x
        self.assertLess(brake["ghost_path"][-1]["cx"], maintain["ghost_path"][-1]["cx"])

    def test_intervention_routes(self):
        tr = _toy_track(1, 100, 200, 60)
        sc = validate_scenario(
            {"intervention": "stronger_braking", "targetTrackIds": [1], "params": {"brakeIntensity": 0.95}}
        )
        g = intervention_to_trajectory(tr, 15, sc)
        self.assertIsNotNone(g)
        self.assertIn("HYPOTHETICAL", g["assumption"])


class TestAggregationAndCache(unittest.TestCase):
    def test_cache_key_stable(self):
        a = cache_key(
            event_id="e1",
            scenario={"intervention": "earlier_braking", "targetTrackIds": [1]},
            personas=["attentive"],
            seed=42,
            model="m",
            n_runs=8,
        )
        b = cache_key(
            event_id="e1",
            scenario={"intervention": "earlier_braking", "targetTrackIds": [1]},
            personas=["attentive"],
            seed=42,
            model="m",
            n_runs=8,
        )
        self.assertEqual(a, b)

    def test_ensemble_deterministic(self):
        analysis = _toy_analysis()
        sc = {
            "targetTrackIds": [1, 2],
            "intervention": "earlier_braking",
            "params": {"delaySec": 0.1, "brakeIntensity": 0.8},
            "horizonSec": 1.0,
            "rationale": "test",
        }
        r1 = run_ensemble(
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            n_runs=3,
            seed=7,
            llm_chat=None,
            use_agents=True,
        )
        # second call hits cache
        r2 = run_ensemble(
            analysis=analysis,
            event=analysis["primary_event"],
            scenario=sc,
            n_runs=3,
            seed=7,
            llm_chat=None,
            use_agents=True,
        )
        self.assertTrue(r1["ok"])
        self.assertTrue(r2["cache_hit"])
        self.assertEqual(r1["summary"]["outcome_counts"], r2["summary"]["outcome_counts"])
        self.assertEqual(len(r1["runs"]), 3)
        # citations: hazard ids resolve
        for run in r1["runs"]:
            for d in run["decisions"]:
                for hid in d.get("perceivedHazardIds") or []:
                    self.assertIn(hid, {1, 2})

    def test_assess_paths(self):
        a = [{"cx": 0, "cy": 0, "t": 0}, {"cx": 10, "cy": 0, "t": 0.1}]
        b = [{"cx": 100, "cy": 0, "t": 0}, {"cx": 110, "cy": 0, "t": 0.1}]
        out = assess_path_interaction(a, b)
        self.assertIn(out["outcome"], {"converge", "separated", "uncertain"})


class TestParser(unittest.TestCase):
    def test_heuristic_brake_earlier(self):
        analysis = _toy_analysis()
        out = parse_whatif(
            "What if the white truck had braked earlier?",
            event=analysis["primary_event"],
            known_track_ids={1, 2},
            llm_chat=None,
        )
        self.assertTrue(out["ok"])
        self.assertEqual(out["scenario"]["intervention"], "earlier_braking")
        self.assertTrue(out["chips"])

    def test_unsupported_swerve(self):
        analysis = _toy_analysis()
        out = parse_whatif(
            "make the driver swerve into the barrier",
            event=analysis["primary_event"],
            known_track_ids={1, 2},
            llm_chat=None,
        )
        self.assertFalse(out["ok"])
        self.assertTrue(out.get("unsupported"))


if __name__ == "__main__":
    unittest.main()
