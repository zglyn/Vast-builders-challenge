"""Multi-agent driver decisions → discrete actions (no coordinates)."""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from persona_config import (
    AGENT_CALL_TIMEOUT_SEC,
    AGENT_CONCURRENCY,
    mid_persona_params,
    sample_persona_params,
)
from scenario_schema import ScenarioValidationError, validate_agent_decision

LLMChat = Callable[[str, str], dict[str, Any]]


def build_perception_package(
    track: dict,
    *,
    event: dict,
    nearby_tracks: list[dict],
    scenario: dict,
    persona_params: dict,
) -> dict[str, Any]:
    """Server-side perception: only what this driver could plausibly know at event time.

    Uses real observed history up to the event; never includes future frames.
    Relative states are image-space.
    """
    path = track.get("full_path") or track.get("path") or []
    t_event = float(event.get("t") or 0)
    past = [p for p in path if p.get("t", 0) <= t_event + 1e-6]
    if not past:
        past = path[:1]

    def state_at(tr: dict) -> dict | None:
        pp = tr.get("full_path") or tr.get("path") or []
        prior = [p for p in pp if p.get("t", 0) <= t_event + 1e-6]
        if not prior:
            return None
        p = prior[-1]
        # velocity from last few points
        vx = vy = 0.0
        if len(prior) >= 2:
            a, b = prior[-2], prior[-1]
            dt = (b["t"] - a["t"]) or 1e-3
            vx = (b["cx"] - a["cx"]) / dt
            vy = (b["cy"] - a["cy"]) / dt
        return {
            "track_id": tr["id"],
            "label": tr.get("label"),
            "cx": p["cx"],
            "cy": p["cy"],
            "vx_px_s": round(vx, 1),
            "vy_px_s": round(vy, 1),
            "units": "image-space pixels",
        }

    self_state = state_at(track)
    relatives = []
    for nt in nearby_tracks:
        if nt["id"] == track["id"]:
            continue
        st = state_at(nt)
        if not st or not self_state:
            continue
        relatives.append(
            {
                "track_id": st["track_id"],
                "label": st["label"],
                "dx_px": round(st["cx"] - self_state["cx"], 1),
                "dy_px": round(st["cy"] - self_state["cy"], 1),
                "rel_dist_px": round(
                    math.hypot(st["cx"] - self_state["cx"], st["cy"] - self_state["cy"]), 1
                ),
                "their_vx_px_s": st["vx_px_s"],
                "their_vy_px_s": st["vy_px_s"],
                "units": "image-space pixels — not meters",
            }
        )

    return {
        "self": self_state,
        "nearby": relatives,
        "observed_trajectory_so_far": [
            {"t": p["t"], "cx": p["cx"], "cy": p["cy"]} for p in past[-12:]
        ],
        "scenario": {
            "intervention": scenario.get("intervention"),
            "params": scenario.get("params"),
            "horizonSec": scenario.get("horizonSec"),
            "label": "HYPOTHETICAL",
        },
        "persona": persona_params,
        "note": "No future observed frames are included. Image-space only.",
    }


def _deterministic_decision(perception: dict, persona_params: dict, scenario: dict) -> dict:
    """Fallback: map persona + scenario to a decision without LLM."""
    yield_w = float(persona_params.get("yield_willingness", 0.5))
    delay = float(persona_params.get("reaction_delay_sec", 0.4))
    brake_i = float(persona_params.get("braking_intensity", 0.6))
    intervention = scenario.get("intervention") or ""

    nearby = perception.get("nearby") or []
    hazard_ids = [n["track_id"] for n in sorted(nearby, key=lambda x: x["rel_dist_px"])[:2]]

    if intervention in ("earlier_braking", "stronger_braking") or yield_w > 0.7:
        action, intensity = "brake", brake_i
    elif intervention in ("later_braking", "delayed_reaction", "weaker_braking"):
        action, intensity = "brake", max(0.2, brake_i * 0.7)
        delay = max(delay, float((scenario.get("params") or {}).get("delay_sec", delay)))
    elif intervention == "aborted_lane_change":
        action, intensity = "abort_lane_change", 0.7
    elif intervention == "delayed_lane_change":
        action, intensity = "steer_left", 0.35
        delay = max(delay, 0.8)
    elif intervention == "maintain_velocity" or yield_w < 0.3:
        action, intensity = "maintain", 0.0
    else:
        action, intensity = "brake" if yield_w >= 0.5 else "maintain", brake_i if yield_w >= 0.5 else 0.0

    return {
        "action": action,
        "intensity": intensity,
        "reactionDelaySec": delay,
        "reasoning": (
            f"Deterministic persona fallback ({persona_params.get('persona')}): "
            f"action={action} given intervention={intervention}."
        ),
        "perceivedHazardIds": hazard_ids,
        "source": "deterministic_fallback",
    }


_AGENT_SYSTEM = (
    "You are ONE vehicle driver in a counterfactual traffic simulation. "
    "Return ONLY JSON: "
    '{"action":"maintain|brake|accelerate|steer_left|steer_right|abort_lane_change",'
    '"intensity":0-1,"reactionDelaySec":number,"reasoning":"1-2 sentences",'
    '"perceivedHazardIds":[int]}. '
    "Use only track IDs present in the perception package. "
    "Do not output coordinates. Do not claim a crash will happen. "
    "Image-space values are not meters."
)


def _call_one_agent(
    *,
    track: dict,
    perception: dict,
    persona_params: dict,
    scenario: dict,
    known_ids: set[int],
    llm_chat: LLMChat | None,
) -> dict[str, Any]:
    decision_raw = None
    source = "deterministic_fallback"
    if llm_chat is not None:
        user = json.dumps(perception)
        try:
            resp = llm_chat(_AGENT_SYSTEM, user)
            if resp.get("ok") and resp.get("text"):
                text = resp["text"]
                fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
                if fence:
                    text = fence.group(1)
                start, end = text.find("{"), text.rfind("}")
                if start >= 0 and end > start:
                    decision_raw = json.loads(text[start : end + 1])
                    source = "wandb"
        except Exception:  # noqa: BLE001
            decision_raw = None

    if decision_raw is None:
        decision_raw = _deterministic_decision(perception, persona_params, scenario)
        source = "deterministic_fallback"

    try:
        decision = validate_agent_decision(decision_raw, known_ids=known_ids)
    except ScenarioValidationError:
        decision = validate_agent_decision(
            _deterministic_decision(perception, persona_params, scenario),
            known_ids=known_ids,
        )
        source = "deterministic_repair"

    decision["source"] = source
    decision["persona"] = persona_params.get("persona")
    decision["persona_params"] = persona_params
    decision["track_id"] = track["id"]
    decision["label"] = track.get("label")
    return decision


def run_driver_agents(
    *,
    tracks_by_id: dict[int, dict],
    agent_ids: list[int],
    event: dict,
    scenario: dict,
    persona_assignment: dict[int, str],
    rng,
    llm_chat: LLMChat | None = None,
    sample: bool = True,
) -> list[dict[str, Any]]:
    """Parallel agent decisions with bounded concurrency."""
    known_ids = set(tracks_by_id.keys())
    nearby = [tracks_by_id[i] for i in agent_ids if i in tracks_by_id]

    jobs = []
    for tid in agent_ids:
        if tid not in tracks_by_id:
            continue
        persona = persona_assignment.get(tid, "attentive")
        params = sample_persona_params(persona, rng) if sample else mid_persona_params(persona)
        perception = build_perception_package(
            tracks_by_id[tid],
            event=event,
            nearby_tracks=nearby,
            scenario=scenario,
            persona_params=params,
        )
        jobs.append((tid, tracks_by_id[tid], perception, params))

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=AGENT_CONCURRENCY) as pool:
        futs = {
            pool.submit(
                _call_one_agent,
                track=tr,
                perception=perc,
                persona_params=params,
                scenario=scenario,
                known_ids=known_ids,
                llm_chat=llm_chat,
            ): tid
            for tid, tr, perc, params in jobs
        }
        for fut in as_completed(futs, timeout=AGENT_CALL_TIMEOUT_SEC * len(jobs) + 5):
            try:
                results.append(fut.result(timeout=AGENT_CALL_TIMEOUT_SEC))
            except Exception as e:  # noqa: BLE001
                tid = futs[fut]
                params = mid_persona_params(persona_assignment.get(tid, "attentive"))
                perc = build_perception_package(
                    tracks_by_id[tid],
                    event=event,
                    nearby_tracks=nearby,
                    scenario=scenario,
                    persona_params=params,
                )
                d = _deterministic_decision(perc, params, scenario)
                d = validate_agent_decision(d, known_ids=known_ids)
                d["source"] = "timeout_fallback"
                d["persona"] = params["persona"]
                d["persona_params"] = params
                d["track_id"] = tid
                d["label"] = tracks_by_id[tid].get("label")
                d["error"] = str(e)[:120]
                results.append(d)
    results.sort(key=lambda d: d.get("track_id") or 0)
    return results
