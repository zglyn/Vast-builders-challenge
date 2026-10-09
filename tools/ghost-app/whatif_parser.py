"""Parse free-form what-if text into a schema-validated scenario via W&B LLM."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from scenario_schema import (
    SUPPORTED_HELP,
    SUPPORTED_INTERVENTIONS,
    ScenarioValidationError,
    scenario_to_chips,
    validate_scenario,
)

ParseLLM = Callable[[str, str], dict[str, Any]]


_SYSTEM = (
    "You convert traffic what-if questions into ONE JSON object only. "
    "No markdown. Schema: "
    '{"targetTrackIds":[int],"intervention":string,"params":object,'
    '"horizonSec":number,"rationale":string}. '
    f"intervention MUST be one of: {', '.join(SUPPORTED_INTERVENTIONS)}. "
    "params may include delaySec (0-2), brakeIntensity (0-1), steerIntensity (0-1). "
    "Only use track IDs from the provided list. "
    "If the request cannot be represented, return "
    '{"unsupported":true,"reason":"..."}. '
    "Never invent collisions or real-world meters."
)


def _extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ScenarioValidationError("Empty model response")
    # strip fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise ScenarioValidationError("Model did not return JSON")


def _match_labeled_target(prompt: str, event: dict) -> list[int]:
    """Pick the vehicle the user named (truck/car/…) so the other stays real footage."""
    p = prompt.lower()
    candidates: list[tuple[int, str]] = []
    if event.get("id_a") is not None:
        candidates.append((int(event["id_a"]), str(event.get("label_a") or "").lower()))
    if event.get("id_b") is not None:
        candidates.append((int(event["id_b"]), str(event.get("label_b") or "").lower()))
    for kw in ("truck", "lorry", "semi", "bus", "van", "suv", "car"):
        if kw not in p:
            continue
        for tid, lab in candidates:
            if kw in lab:
                return [tid]
    for kw in ("white", "black", "red", "blue"):
        if kw not in p:
            continue
        for tid, lab in candidates:
            if kw in lab:
                return [tid]
    return []


def _default_single_actor(event: dict) -> list[int]:
    """Primary actor only — leave the other vehicle as sharp OBSERVED pixels."""
    if event.get("id_a") is not None:
        return [int(event["id_a"])]
    if event.get("id_b") is not None:
        return [int(event["id_b"])]
    return []


def _heuristic_parse(prompt: str, event: dict) -> dict[str, Any]:
    """Deterministic fallback when LLM unavailable — maps keywords to interventions."""
    p = prompt.lower()

    # Creative/comic asks are allowed as labeled hypotheticals (export adds gags).
    # Only reject clearly impossible non-traffic asks.
    for bad in ("kill", "fly", "teleport", "explode the city"):
        if bad in p:
            raise ScenarioValidationError(
                f"That request is not supported by the ghost engine. {SUPPORTED_HELP}",
                unsupported=True,
            )

    move_everyone = any(
        w in p for w in ("everyone", "both", "all vehicles", "all cars", "cautious")
    )

    intervention = "maintain_velocity"
    params: dict[str, Any] = {}
    if "distract" in p:
        intervention = "delayed_reaction"
        params = {"delaySec": 1.0, "brakeIntensity": 0.4}
    elif "earlier" in p and "brake" in p:
        intervention = "earlier_braking"
        params = {"delaySec": 0.1, "brakeIntensity": 0.8}
    elif "later" in p and "brake" in p:
        intervention = "later_braking"
        params = {"delaySec": 0.8, "brakeIntensity": 0.7}
    elif ("stronger" in p or "harder" in p or "slam" in p) and (
        "brake" in p or "break" in p
    ):
        intervention = "stronger_braking"
        params = {"brakeIntensity": 0.95, "delaySec": 0.25}
    elif ("weaker" in p or "lighter" in p) and ("brake" in p or "break" in p):
        intervention = "weaker_braking"
        params = {"brakeIntensity": 0.3, "delaySec": 0.3}
    elif "brake" in p or "brak" in p or "breaks" in p or "broke" in p:
        intervention = "earlier_braking"
        params = {"delaySec": 0.15, "brakeIntensity": 0.75}
    elif "wait" in p or "delayed lane" in p or "lane change" in p and "wait" in p:
        intervention = "delayed_lane_change"
        params = {"delaySec": 2.0, "steerIntensity": 0.4}
    elif "abort" in p and "lane" in p:
        intervention = "aborted_lane_change"
        params = {"steerIntensity": 0.6, "delaySec": 0.2}
    elif "cautious" in p or "everyone" in p:
        intervention = "stronger_braking"
        params = {"brakeIntensity": 0.9, "delaySec": 0.2}
        move_everyone = True
    elif "exclude" in p or "remove" in p:
        intervention = "exclude_vehicle"
    elif "heading" in p:
        intervention = "maintain_heading"
    elif "swerve" in p or "spin" in p:
        # Creative comic maneuver — maps to a steer intervention (not a real crash claim)
        intervention = "delayed_lane_change"
        params = {"delaySec": 0.1, "steerIntensity": 0.85}
    elif "crash" in p or "collide" in p or "smash" in p or "boom" in p:
        # Storytelling prompt: strong brake + comic export overlay handles the gag
        intervention = "stronger_braking"
        params = {"brakeIntensity": 0.95, "delaySec": 0.05}

    # Mentions of track numbers win
    targets: list[int] = []
    for m in re.finditer(r"(?:track|#)\s*(\d+)", p):
        tid = int(m.group(1))
        if tid not in targets:
            targets.append(tid)

    if not targets:
        if move_everyone:
            if event.get("id_a") is not None:
                targets.append(int(event["id_a"]))
            if event.get("id_b") is not None:
                targets.append(int(event["id_b"]))
        else:
            # Single actor: named vehicle, else primary — leave the other car sharp
            named = _match_labeled_target(p, event)
            targets = named[:1] if named else _default_single_actor(event)

    return {
        "targetTrackIds": targets,
        "intervention": intervention,
        "params": params,
        "horizonSec": 1.5,
        "rationale": (
            f"Heuristic parse (no LLM). Mapped to {intervention}; "
            f"moving track(s) {targets} only — others stay OBSERVED."
        ),
    }


def parse_whatif(
    prompt: str,
    *,
    event: dict,
    known_track_ids: set[int],
    llm_chat: ParseLLM | None = None,
) -> dict[str, Any]:
    """Return {ok, scenario, chips, source} or {ok:False, unsupported, message}."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "unsupported": False, "message": "Enter a what-if question."}

    track_list = sorted(known_track_ids)
    event_blurb = json.dumps(
        {
            "kind": event.get("kind"),
            "t": event.get("t"),
            "id_a": event.get("id_a"),
            "id_b": event.get("id_b"),
            "label_a": event.get("label_a"),
            "label_b": event.get("label_b"),
        }
    )
    user = (
        f"Known track IDs: {track_list}\n"
        f"Selected event: {event_blurb}\n"
        f"What-if: {prompt}"
    )

    raw: dict[str, Any] | None = None
    source = "heuristic"
    if llm_chat is not None:
        try:
            resp = llm_chat(_SYSTEM, user)
            if resp.get("ok") and resp.get("text"):
                raw = _extract_json(resp["text"])
                source = "wandb"
        except ScenarioValidationError:
            raw = None
        except Exception:  # noqa: BLE001
            raw = None

    if raw is None:
        try:
            raw = _heuristic_parse(prompt, event)
            source = "heuristic"
        except ScenarioValidationError as e:
            return {
                "ok": False,
                "unsupported": e.unsupported,
                "message": str(e),
                "supported": list(SUPPORTED_INTERVENTIONS),
            }

    if raw.get("unsupported"):
        return {
            "ok": False,
            "unsupported": True,
            "message": str(raw.get("reason") or SUPPORTED_HELP),
            "supported": list(SUPPORTED_INTERVENTIONS),
        }

    # Default targets from event if model omitted them
    if not raw.get("targetTrackIds") and not raw.get("target_track_ids"):
        targets = []
        if event.get("id_a") is not None:
            targets.append(int(event["id_a"]))
        if event.get("id_b") is not None:
            targets.append(int(event["id_b"]))
        raw["targetTrackIds"] = targets

    try:
        scenario = validate_scenario(raw, known_track_ids=known_track_ids)
        # If LLM returned both event vehicles for a single-actor ask, narrow to one
        # so the lead car is not erased/re-pasted (that was washing it out).
        prompt_l = prompt.lower()
        move_everyone = any(
            w in prompt_l for w in ("everyone", "both", "all vehicles", "all cars", "cautious")
        )
        single_actor = scenario["intervention"] in {
            "earlier_braking",
            "later_braking",
            "stronger_braking",
            "weaker_braking",
            "delayed_reaction",
            "delayed_lane_change",
            "aborted_lane_change",
            "maintain_heading",
            "maintain_velocity",
        }
        if (
            single_actor
            and not move_everyone
            and len(scenario.get("targetTrackIds") or []) > 1
            and not re.search(r"(?:track|#)\s*\d+", prompt_l)
        ):
            named = _match_labeled_target(prompt_l, event)
            scenario["targetTrackIds"] = named[:1] if named else scenario["targetTrackIds"][:1]
            scenario["rationale"] = (
                (scenario.get("rationale") or "")
                + f" [narrowed to track {scenario['targetTrackIds']} so other vehicles stay OBSERVED]"
            )
    except ScenarioValidationError as e:
        if e.unsupported:
            return {
                "ok": False,
                "unsupported": True,
                "message": str(e),
                "supported": list(SUPPORTED_INTERVENTIONS),
            }
        # Fall back to heuristic if LLM schema failed
        try:
            scenario = validate_scenario(_heuristic_parse(prompt, event), known_track_ids=known_track_ids)
            source = "heuristic_repair"
        except ScenarioValidationError as e2:
            return {
                "ok": False,
                "unsupported": e2.unsupported,
                "message": str(e2),
                "supported": list(SUPPORTED_INTERVENTIONS),
            }

    return {
        "ok": True,
        "scenario": scenario,
        "chips": scenario_to_chips(scenario),
        "source": source,
        "notice": "Parsed scenario is HYPOTHETICAL. Confirm chips before running.",
    }
