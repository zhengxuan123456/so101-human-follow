"""Per-user gaze state with project isolation and atomic replacement.

Runtime state need not inherit the source checkout's Windows ACLs. Callers can
pass any explicit path to read/write, or override the data root for isolated tests.
This module never migrates, deletes, or changes permissions on legacy repo data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def gaze_state_path(project_path: Path, *, data_root: Path | None = None) -> Path:
    """Return a stable namespace for this resolved project, never its source dir."""
    project = os.path.normcase(str(Path(project_path).expanduser().resolve()))
    namespace = hashlib.sha256(project.encode("utf-8")).hexdigest()[:16]
    if data_root is None:
        local_app_data = os.environ.get("LOCALAPPDATA") if os.name == "nt" else None
        data_root = Path(local_app_data) if local_app_data else Path.home() / ".local" / "share"
    return Path(data_root) / "so101-human-follow" / namespace / "gaze_calibration.json"


def atomic_write_gaze(path: Path, payload: dict[str, Any]) -> None:
    """Commit finite JSON atomically; failed cleanup cannot hide the cause."""
    if not isinstance(payload, dict):
        raise ValueError("Gaze state must be a JSON object.")
    # Reject invalid values before creating a temp file or touching old state.
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=".gaze-", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8", newline="\n")
        descriptor = None
        with stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                LOGGER.warning("Could not close gaze temporary descriptor", exc_info=True)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Could not remove gaze temporary file: %s", temporary, exc_info=True)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Gaze state contains a nonfinite number.")
    return parsed


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Gaze state contains a nonfinite constant: {value}.")


def read_gaze(path: Path) -> dict[str, Any] | None:
    """Read state, distinguishing absent data from corrupt or inaccessible data."""
    path = Path(path)
    try:
        serialized = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except UnicodeError as exc:
        raise ValueError("Gaze state is not valid UTF-8.") from exc
    try:
        payload = json.loads(serialized, parse_float=_finite_float, parse_constant=_invalid_constant)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid gaze state JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Gaze state must be a JSON object.")
    return payload
