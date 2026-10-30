from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .config import ALL_JOINTS, CONTROLLED_JOINTS, FIXED_JOINTS


SCHEMA_VERSION = 1
MIN_CONTROLLED_SPAN = 2.0
MIDPOINT_EDGE_MARGIN = 0.25


@dataclass(frozen=True)
class SafeRangeProfile:
    """User-selected working envelope, separate from LeRobot motor calibration."""

    robot_id: str
    calibration_fingerprint: str
    controlled_limits: dict[str, tuple[float, float]]
    fixed_positions: dict[str, float]
    startup_positions: dict[str, float]
    created_at: str

    @classmethod
    def create(
        cls,
        *,
        robot_id: str,
        calibration_fingerprint: str,
        controlled_limits: Mapping[str, tuple[float, float]],
        fixed_positions: Mapping[str, float],
        startup_positions: Mapping[str, float],
    ) -> "SafeRangeProfile":
        profile = cls(
            robot_id=str(robot_id),
            calibration_fingerprint=str(calibration_fingerprint),
            controlled_limits={
                str(joint): (float(bounds[0]), float(bounds[1]))
                for joint, bounds in controlled_limits.items()
            },
            fixed_positions={str(joint): float(value) for joint, value in fixed_positions.items()},
            startup_positions={
                str(joint): float(value) for joint, value in startup_positions.items()
            },
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        profile.validate()
        return profile

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SafeRangeProfile":
        if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Unsupported safe-range profile schema version.")
        limits_raw = raw.get("controlled_limits")
        if not isinstance(limits_raw, Mapping):
            raise ValueError("Safe-range profile is missing controlled_limits.")
        fixed_raw = raw.get("fixed_positions")
        startup_raw = raw.get("startup_positions")
        if not isinstance(fixed_raw, Mapping) or not isinstance(startup_raw, Mapping):
            raise ValueError("Safe-range profile is missing fixed/startup positions.")
        profile = cls(
            robot_id=str(raw.get("robot_id", "")),
            calibration_fingerprint=str(raw.get("calibration_fingerprint", "")),
            controlled_limits={
                str(joint): (float(bounds[0]), float(bounds[1]))
                for joint, bounds in limits_raw.items()
            },
            fixed_positions={str(joint): float(value) for joint, value in fixed_raw.items()},
            startup_positions={str(joint): float(value) for joint, value in startup_raw.items()},
            created_at=str(raw.get("created_at", "")),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.robot_id or not self.calibration_fingerprint:
            raise ValueError("Safe-range profile must identify the robot calibration.")
        if set(self.controlled_limits) != set(CONTROLLED_JOINTS):
            raise ValueError("Safe-range limits must contain exactly ID3-ID6.")
        if set(self.fixed_positions) != set(FIXED_JOINTS):
            raise ValueError("Fixed positions must contain exactly ID1-ID2.")
        if set(self.startup_positions) != set(ALL_JOINTS):
            raise ValueError("Startup positions must contain all six joints.")
        if not self.created_at:
            raise ValueError("Safe-range profile is missing its creation time.")

        for joint, (low, high) in self.controlled_limits.items():
            startup = self.startup_positions[joint]
            if not all(math.isfinite(value) for value in (low, high, startup)):
                raise ValueError(f"Non-finite safe-range value for {joint}.")
            if high - low < MIN_CONTROLLED_SPAN:
                raise ValueError(
                    f"Safe range for {joint} is too small; move it through at least "
                    f"{MIN_CONTROLLED_SPAN:.1f} position units."
                )
            if not low + MIDPOINT_EDGE_MARGIN < startup < high - MIDPOINT_EDGE_MARGIN:
                raise ValueError(
                    f"Return {joint} to a comfortable middle position before saving."
                )

        values = [*self.fixed_positions.values(), *self.startup_positions.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Safe-range profile contains a non-finite position.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "robot_id": self.robot_id,
            "calibration_fingerprint": self.calibration_fingerprint,
            "controlled_limits": {
                joint: [low, high] for joint, (low, high) in self.controlled_limits.items()
            },
            "fixed_positions": dict(self.fixed_positions),
            "startup_positions": dict(self.startup_positions),
            "created_at": self.created_at,
        }


def load_safe_range_profile(path: str | Path) -> SafeRangeProfile | None:
    profile_path = Path(path)
    if not profile_path.is_file():
        return None
    with profile_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError("Safe-range profile root must be an object.")
    return SafeRangeProfile.from_dict(raw)


def save_safe_range_profile(path: str | Path, profile: SafeRangeProfile) -> None:
    """Atomically persist a validated profile in the destination directory."""

    profile.validate()
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{profile_path.name}.", suffix=".tmp", dir=profile_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(profile.to_dict(), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, profile_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
