"""Schema-validated what-if scenario objects (Python stand-in for Zod)."""

from __future__ import annotations

from typing import Any

SUPPORTED_INTERVENTIONS = (
    "delayed_reaction",
    "earlier_braking",
    "later_braking",
    "stronger_braking",
    "weaker_braking",
    "maintain_velocity",
    "maintain_heading",
    "delayed_lane_change",
    "aborted_lane_change",
    "exclude_vehicle",
)

SUPPORTED_ACTIONS = (
    "maintain",
    "brake",
    "accelerate",
    "steer_left",
    "steer_right",
    "abort_lane_change",
)

SUPPORTED_HELP = (
    "Supported interventions: delayed reaction; earlier/later/stronger/weaker braking; "
    "maintain velocity; maintain heading; delayed or aborted lane change; exclude vehicle. "
    "Ask about vehicles already tracked in this clip (use track IDs from the event list)."
)


class ScenarioValidationError(ValueError):
    def __init__(self, message: str, *, unsupported: bool = False):
        super().__init__(message)
        self.unsupported = unsupported


def _as_int_list(val: Any) -> list[int]:
    if val is None:
        return []
    if not isinstance(val, list):
        raise ScenarioValidationError("targetTrackIds must be a list of integers")
    out: list[int] = []
    for x in val:
        try:
            out.append(int(x))
        except (TypeError, ValueError) as e:
            raise ScenarioValidationError(f"invalid track id: {x!r}") from e
    return out


def validate_scenario(raw: dict[str, Any], *, known_track_ids: set[int] | None = None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ScenarioValidationError("scenario must be an object")

    intervention = str(raw.get("intervention") or "").strip()
    if intervention not in SUPPORTED_INTERVENTIONS:
        raise ScenarioValidationError(
            f"Unsupported intervention {intervention!r}. {SUPPORTED_HELP}",
            unsupported=True,
        )

    targets = _as_int_list(raw.get("targetTrackIds") or raw.get("target_track_ids"))
    if not targets and intervention != "exclude_vehicle":
        # allow empty only if caller will fill from primary event
        pass

    if known_track_ids is not None and targets:
        unknown = [t for t in targets if t not in known_track_ids]
        if unknown:
            raise ScenarioValidationError(
                f"Unknown track IDs {unknown}. Known tracks: {sorted(known_track_ids)}. "
                "IDs are segment-local."
            )

    params = raw.get("params") or {}
    if not isinstance(params, dict):
        raise ScenarioValidationError("params must be an object")

    horizon = raw.get("horizonSec", raw.get("horizon_sec", 1.5))
    try:
        horizon_sec = float(horizon)
    except (TypeError, ValueError) as e:
        raise ScenarioValidationError("horizonSec must be a number") from e
    horizon_sec = max(0.5, min(3.0, horizon_sec))

    # Normalize optional numeric params
    clean_params: dict[str, Any] = {}
    for key in ("delaySec", "delay_sec", "brakeIntensity", "brake_intensity", "steerIntensity", "steer_intensity"):
        if key in params:
            try:
                clean_params[key] = float(params[key])
            except (TypeError, ValueError) as e:
                raise ScenarioValidationError(f"params.{key} must be numeric") from e

    # Canonical snake aliases
    if "delaySec" in clean_params and "delay_sec" not in clean_params:
        clean_params["delay_sec"] = clean_params["delaySec"]
    if "brakeIntensity" in clean_params and "brake_intensity" not in clean_params:
        clean_params["brake_intensity"] = clean_params["brakeIntensity"]
    if "steerIntensity" in clean_params and "steer_intensity" not in clean_params:
        clean_params["steer_intensity"] = clean_params["steerIntensity"]

    rationale = str(raw.get("rationale") or "").strip()
    if len(rationale) > 400:
        rationale = rationale[:400]

    return {
        "targetTrackIds": targets,
        "intervention": intervention,
        "params": clean_params,
        "horizonSec": horizon_sec,
        "rationale": rationale,
        "label": "HYPOTHETICAL",
    }


def validate_agent_decision(raw: dict[str, Any], *, known_ids: set[int]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ScenarioValidationError("agent decision must be an object")
    action = str(raw.get("action") or "").strip()
    if action not in SUPPORTED_ACTIONS:
        raise ScenarioValidationError(f"unsupported action {action!r}")

    try:
        intensity = float(raw.get("intensity", 0.5))
    except (TypeError, ValueError) as e:
        raise ScenarioValidationError("intensity must be 0-1") from e
    intensity = max(0.0, min(1.0, intensity))

    try:
        delay = float(raw.get("reactionDelaySec", raw.get("reaction_delay_sec", 0.3)))
    except (TypeError, ValueError) as e:
        raise ScenarioValidationError("reactionDelaySec must be numeric") from e
    delay = max(0.0, min(2.0, delay))

    reasoning = str(raw.get("reasoning") or "").strip()
    if len(reasoning) > 280:
        reasoning = reasoning[:280]

    hazards = raw.get("perceivedHazardIds") or raw.get("perceived_hazard_ids") or []
    if not isinstance(hazards, list):
        hazards = []
    hazard_ids: list[int] = []
    for h in hazards:
        try:
            hid = int(h)
        except (TypeError, ValueError):
            continue
        if hid in known_ids:
            hazard_ids.append(hid)

    # Strip invented IDs from reasoning is soft — we keep text but flag unknown
    return {
        "action": action,
        "intensity": intensity,
        "reactionDelaySec": delay,
        "reasoning": reasoning,
        "perceivedHazardIds": hazard_ids,
    }


def scenario_to_chips(scenario: dict[str, Any]) -> list[dict[str, str]]:
    chips = [
        {"key": "intervention", "label": "intervention", "value": scenario["intervention"]},
        {
            "key": "targets",
            "label": "targets",
            "value": ",".join(str(t) for t in scenario.get("targetTrackIds") or []) or "(primary event)",
        },
        {"key": "horizon", "label": "horizonSec", "value": str(scenario.get("horizonSec", 1.5))},
    ]
    params = scenario.get("params") or {}
    # Prefer camelCase chip keys; skip snake aliases that duplicate them.
    shown = set()
    for k, v in params.items():
        if not (k.endswith("Sec") or k.endswith("_sec") or "intensity" in k.lower() or "Intensity" in k):
            continue
        canon = (
            "delaySec" if k in ("delaySec", "delay_sec") else
            "brakeIntensity" if k in ("brakeIntensity", "brake_intensity") else
            "steerIntensity" if k in ("steerIntensity", "steer_intensity") else
            k
        )
        if canon in shown:
            continue
        shown.add(canon)
        chips.append({"key": f"param:{canon}", "label": canon, "value": str(v)})
    if scenario.get("rationale"):
        chips.append({"key": "rationale", "label": "rationale", "value": scenario["rationale"][:80]})
    return chips
