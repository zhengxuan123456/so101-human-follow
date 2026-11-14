"""Position progress for timed moves between two joint targets."""

import math


# At halfway, cosine progress changes pi/2 times faster than linear progress.
COSINE_PEAK_SPEED_FACTOR = math.pi / 2


def cosine_progress(phase: float) -> float:
    """Map a normalized time in [0, 1] to position with zero endpoint slopes."""
    return (1.0 - math.cos(math.pi * phase)) / 2.0
