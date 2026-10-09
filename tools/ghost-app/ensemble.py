"""Ensemble multi-agent what-if runs with caching and aggregation."""

from __future__ import annotations

import hashlib
import json
import random
import threading
from typing import Any, Callable

from driver_agents import run_driver_agents
from ghost_engine import (
    Track,
    assess_path_interaction,
    decision_to_trajectory,
    intervention_to_trajectory,
    nearby_track_ids,
)
from persona_config import (
    DEFAULT_ENSEMBLE_RUNS,
    MAX_AGENTS_PER_RUN,
    MAX_ENSEMBLE_RUNS,
    PERSONA_NAMES,
)
from scenario_schema import validate_scenario

LLMChat = Callable[[str, str], dict[str, Any]]

_cache: dict[str, dict[str, Any]] = {}
_cache_lock = threading.Lock()


def cache_key(
    *,
    event_id: str,
    scenario: dict,
    personas: list[str],
    seed: int,
    model: str,
    n_runs: int,
) -> str:
    payload = {
        "event_id": event_id,
        "scenario": scenario,
        "personas": personas,
        "seed": seed,
        "model": model,
        "n_runs": n_runs,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _track_from_dict(td: dict) -> Track:
    tr = Track(id=int(td["id"]), label=td.get("label") or "car", bbox=[0, 0, 1, 1], conf=1.0)
    for p in td.get("full_path") or td.get("path") or []:
        tr.history.append(
            {
                "frame": int(p["frame"]),
                "t": float(p["t"]),
                "cx": float(p["cx"]),
                "cy": float(p["cy"]),
                "bbox": list(p.get("bbox") or [p["cx"] - 10, p["cy"] - 10, p["cx"] + 10, p["cy"] + 10]),
                "conf": 1.0,
            }
        )
        if p.get("bbox"):
            tr.bbox = list(p["bbox"])
    tr.hits = len(tr.history)
    return tr


def _assign_personas(agent_ids: list[int], rng: random.Random, forced: list[str] | None) -> dict[int, str]:
    out = {}
    for i, tid in enumerate(agent_ids):
        if forced and i < len(forced):
            out[tid] = forced[i] if forced[i] in PERSONA_NAMES else "attentive"
        else:
            out[tid] = rng.choice(list(PERSONA_NAMES))
    return out


def run_ensemble(
    *,
    analysis: dict,
    event: dict,
    scenario: dict,
    n_runs: int = DEFAULT_ENSEMBLE_RUNS,
    seed: int = 42,
    personas: list[str] | None = None,
    llm_chat: LLMChat | None = None,
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
    use_agents: bool = True,
) -> dict[str, Any]:
    scenario = validate_scenario(
        scenario,
        known_track_ids={int(t["id"]) for t in analysis.get("tracks") or []},
    )
    n_runs = max(1, min(MAX_ENSEMBLE_RUNS, int(n_runs)))
    fps = float(analysis.get("fps") or 30.0)
    tracks_dicts = {int(t["id"]): t for t in analysis.get("tracks") or []}
    track_objs = {tid: _track_from_dict(td) for tid, td in tracks_dicts.items()}

    targets = list(scenario.get("targetTrackIds") or [])
    if not targets:
        if event.get("id_a") is not None:
            targets.append(int(event["id_a"]))
        if event.get("id_b") is not None:
            targets.append(int(event["id_b"]))
    scenario["targetTrackIds"] = targets

    # Build Track objects list for nearby helper
    track_list = list(track_objs.values())
    near = nearby_track_ids(track_list, event, max_nearby=max(0, MAX_AGENTS_PER_RUN - len(targets)))
    agent_ids = []
    for tid in targets + near:
        if tid in tracks_dicts and tid not in agent_ids:
            agent_ids.append(tid)
    agent_ids = agent_ids[:MAX_AGENTS_PER_RUN]

    event_id = f"{event.get('kind')}:{event.get('frame')}:{event.get('id_a')}:{event.get('id_b')}"
    key = cache_key(
        event_id=event_id,
        scenario=scenario,
        personas=personas or [],
        seed=seed,
        model=model_name,
        n_runs=n_runs,
    )
    with _cache_lock:
        if key in _cache:
            cached = dict(_cache[key])
            cached["cache_hit"] = True
            return cached

    progress: list[str] = ["ensemble:start"]
    runs_out: list[dict[str, Any]] = []
    master = random.Random(seed)

    for run_i in range(n_runs):
        run_seed = master.randint(0, 2**31 - 1)
        rng = random.Random(run_seed)
        assignment = _assign_personas(agent_ids, rng, personas)
        progress.append(f"run {run_i+1}/{n_runs}: agents deciding (seed={run_seed})")

        if use_agents:
            decisions = run_driver_agents(
                tracks_by_id=tracks_dicts,
                agent_ids=agent_ids,
                event=event,
                scenario=scenario,
                persona_assignment=assignment,
                rng=rng,
                llm_chat=llm_chat,
                sample=True,
            )
        else:
            decisions = []

        ghosts = []
        # Target vehicles: apply agent decision if present, else intervention
        decision_by_id = {d["track_id"]: d for d in decisions}
        for tid in agent_ids:
            tr = track_objs.get(tid)
            if not tr or not tr.history:
                continue
            if tid in decision_by_id:
                g = decision_to_trajectory(
                    tr,
                    int(event["frame"]),
                    decision_by_id[tid],
                    horizon_sec=float(scenario.get("horizonSec") or 1.5),
                    fps=fps,
                )
            elif tid in targets:
                g = intervention_to_trajectory(tr, int(event["frame"]), scenario, fps=fps)
            else:
                # baseline constant velocity via empty maintain decision
                g = decision_to_trajectory(
                    tr,
                    int(event["frame"]),
                    {"action": "maintain", "intensity": 0.0, "reactionDelaySec": 0.0},
                    horizon_sec=float(scenario.get("horizonSec") or 1.5),
                    fps=fps,
                )
            if g:
                ghosts.append(g)

        # Interaction assessment between first two ghosts with paths
        paths = [g for g in ghosts if g.get("ghost_path")]
        if len(paths) >= 2:
            assessment = assess_path_interaction(paths[0]["ghost_path"], paths[1]["ghost_path"])
        elif len(targets) >= 2 and all(t in track_objs for t in targets[:2]):
            # try intervention-only on the two targets
            ga = intervention_to_trajectory(track_objs[targets[0]], int(event["frame"]), scenario, fps=fps)
            gb = intervention_to_trajectory(track_objs[targets[1]], int(event["frame"]), scenario, fps=fps)
            if ga and gb and ga.get("ghost_path") and gb.get("ghost_path"):
                assessment = assess_path_interaction(ga["ghost_path"], gb["ghost_path"])
            else:
                assessment = {"outcome": "uncertain", "min_norm_dist": None}
        else:
            assessment = {
                "outcome": "uncertain",
                "min_norm_dist": None,
                "note": "Need two trajectories to assess interaction",
            }

        runs_out.append(
            {
                "run_index": run_i,
                "seed": run_seed,
                "persona_assignment": {str(k): v for k, v in assignment.items()},
                "decisions": decisions,
                "ghosts": ghosts,
                "assessment": assessment,
            }
        )
        progress.append(f"run {run_i+1}/{n_runs}: outcome={assessment.get('outcome')}")

    # Aggregate
    counts = {"converge": 0, "separated": 0, "uncertain": 0}
    for r in runs_out:
        oc = (r.get("assessment") or {}).get("outcome") or "uncertain"
        counts[oc] = counts.get(oc, 0) + 1

    # Pick most / least interacting by min_norm_dist
    scored = []
    for r in runs_out:
        nd = (r.get("assessment") or {}).get("min_norm_dist")
        scored.append((nd if nd is not None else 99.0, r["run_index"]))
    scored.sort()
    most_idx = scored[0][1] if scored else 0
    least_idx = scored[-1][1] if scored else 0

    # Ghost fan: all paths per track across runs
    fan: dict[str, list] = {}
    for r in runs_out:
        for g in r.get("ghosts") or []:
            tid = str(g["track_id"])
            fan.setdefault(tid, []).append(
                {
                    "run_index": r["run_index"],
                    "path": g.get("ghost_path") or [],
                    "persona": (r.get("persona_assignment") or {}).get(tid),
                }
            )

    summary = {
        "n_runs": n_runs,
        "outcome_counts": counts,
        "outcome_share_text": (
            f"{counts.get('separated', 0)} of {n_runs} simulated runs: paths separated; "
            f"{counts.get('converge', 0)} converge (image-space); "
            f"{counts.get('uncertain', 0)} uncertain"
        ),
        "most_interacting_run": most_idx,
        "least_interacting_run": least_idx,
        "disclaimer": (
            "Simulated runs under assumed driver profiles. "
            "Not real-world probabilities, not a prediction, not a finding of fault."
        ),
    }

    result = {
        "ok": True,
        "cache_hit": False,
        "cache_key": key,
        "scenario": scenario,
        "event": {
            "kind": event.get("kind"),
            "t": event.get("t"),
            "frame": event.get("frame"),
            "id_a": event.get("id_a"),
            "id_b": event.get("id_b"),
        },
        "agent_ids": agent_ids,
        "runs": runs_out,
        "fan": fan,
        "summary": summary,
        "progress": progress,
        "label": "HYPOTHETICAL",
        "notice": (
            "Simulated driver behavior under assumed profiles. "
            "Not a prediction or a finding of fault."
        ),
    }
    with _cache_lock:
        _cache[key] = result
    return result
