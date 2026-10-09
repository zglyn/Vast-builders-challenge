"""Documented driver persona → numeric parameter ranges.

Personas are labeled assumptions for counterfactual simulation only.
They are NOT claims about real drivers.
"""

from __future__ import annotations

from typing import Any

# Bounds clamp agent output into plausible image-space control ranges.
# Intensities are dimensionless 0–1 multipliers on deterministic ghost kinematics.
PERSONA_PARAMS: dict[str, dict[str, Any]] = {
    "attentive": {
        "reaction_delay_sec": (0.15, 0.45),
        "braking_intensity": (0.45, 0.85),
        "yield_willingness": 0.75,
        "description": "Typical alert driver; moderate delay, willing to brake/yield.",
    },
    "cautious": {
        "reaction_delay_sec": (0.10, 0.35),
        "braking_intensity": (0.65, 1.0),
        "yield_willingness": 0.95,
        "description": "Earlier/stronger braking; high willingness to yield.",
    },
    "distracted": {
        "reaction_delay_sec": (0.6, 1.4),
        "braking_intensity": (0.2, 0.55),
        "yield_willingness": 0.35,
        "description": "Delayed reaction; weaker braking response.",
    },
    "aggressive": {
        "reaction_delay_sec": (0.2, 0.5),
        "braking_intensity": (0.15, 0.45),
        "yield_willingness": 0.2,
        "description": "Short delay but low yield; prefers maintain/accelerate.",
    },
    "hesitant": {
        "reaction_delay_sec": (0.4, 0.9),
        "braking_intensity": (0.4, 0.7),
        "yield_willingness": 0.85,
        "description": "Delayed but then tends to brake/abort maneuvers.",
    },
}

PERSONA_NAMES = tuple(PERSONA_PARAMS.keys())

# Global kinematic clamps (image-space, relative to observed speed).
MAX_DECEL_FRAC = 1.0  # full stop of image-space velocity component along heading
MAX_ACCEL_FRAC = 0.35
MAX_LATERAL_RATE_FRAC = 0.45  # fraction of speed diverted laterally per second
MAX_REACTION_DELAY_SEC = 2.0
MIN_REACTION_DELAY_SEC = 0.0
DEFAULT_ENSEMBLE_RUNS = 8
MAX_ENSEMBLE_RUNS = 12
MAX_AGENTS_PER_RUN = 6
AGENT_CALL_TIMEOUT_SEC = 25
AGENT_CONCURRENCY = 4


def sample_persona_params(persona: str, rng) -> dict[str, float]:
    """Sample numeric params for a persona using a seeded RNG."""
    if persona not in PERSONA_PARAMS:
        raise ValueError(f"unknown persona: {persona}")
    cfg = PERSONA_PARAMS[persona]
    rd0, rd1 = cfg["reaction_delay_sec"]
    bi0, bi1 = cfg["braking_intensity"]
    return {
        "persona": persona,
        "reaction_delay_sec": float(rng.uniform(rd0, rd1)),
        "braking_intensity": float(rng.uniform(bi0, bi1)),
        "yield_willingness": float(cfg["yield_willingness"]),
    }


def mid_persona_params(persona: str) -> dict[str, float]:
    """Deterministic mid-range params (fallback when LLM unavailable)."""
    if persona not in PERSONA_PARAMS:
        raise ValueError(f"unknown persona: {persona}")
    cfg = PERSONA_PARAMS[persona]
    rd0, rd1 = cfg["reaction_delay_sec"]
    bi0, bi1 = cfg["braking_intensity"]
    return {
        "persona": persona,
        "reaction_delay_sec": (rd0 + rd1) / 2.0,
        "braking_intensity": (bi0 + bi1) / 2.0,
        "yield_willingness": float(cfg["yield_willingness"]),
    }
