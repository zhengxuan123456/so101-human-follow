from __future__ import annotations

from typing import Mapping

import numpy as np


class RerunViewer:
    def __init__(self, *, enabled: bool, memory_limit: str):
        self.enabled = enabled
        self.rr = None
        if not enabled:
            return
        try:
            import rerun as rr

            rr.init("so101_human_follow")
            rr.spawn(memory_limit=memory_limit)
            self.rr = rr
        except Exception as exc:
            self.enabled = False
            print(f"Rerun viewer unavailable; continuing in terminal-only mode: {exc}")

    def log(
        self,
        frame_rgb: np.ndarray,
        *,
        action: Mapping[str, float],
        armed: bool,
        pose_valid: bool,
    ) -> None:
        if not self.enabled or self.rr is None:
            return
        try:
            self.rr.log("camera/front", self.rr.Image(frame_rgb))
            self.rr.log("state/armed", self.rr.Scalar(1.0 if armed else 0.0))
            self.rr.log("state/pose_valid", self.rr.Scalar(1.0 if pose_valid else 0.0))
            for joint, value in action.items():
                self.rr.log(f"action/{joint}", self.rr.Scalar(float(value)))
        except Exception as exc:
            self.enabled = False
            print(f"Rerun logging failed and was disabled: {exc}")

    def close(self) -> None:
        if self.rr is not None and hasattr(self.rr, "rerun_shutdown"):
            try:
                self.rr.rerun_shutdown()
            except Exception as exc:
                print(f"Rerun shutdown warning: {exc}")

