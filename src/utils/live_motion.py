"""Bounded, time-aware live motion processing (no additional inference model)."""
import time
import numpy as np


class AdaptiveMotionFilter:
    """Suppress small measurement jitter while responding faster to expressions."""
    def __init__(self):
        self.previous = {}
        self.timestamp = None

    def process(self, info, reset=False, now=None):
        now = time.monotonic() if now is None else now
        dt = np.clip(now - self.timestamp, .001, .25) if self.timestamp is not None else .033
        self.timestamp = now
        if reset:
            self.previous.clear()
        result = dict(info)
        for key in ("exp", "pitch", "yaw", "roll"):
            value = np.asarray(info[key], dtype=np.float32)
            if key in self.previous:
                previous = self.previous[key]
                speed = np.abs(value - previous) / dt
                # Expressions are in canonical-coordinate units; pose is in degrees.
                cutoff = 2.5 + (35.0 if key == "exp" else .12) * speed
                alpha = 1 - np.exp(-2 * np.pi * cutoff * dt)
                value = previous + alpha * (value - previous)
            self.previous[key] = value.copy()
            result[key] = value
        return result


def bounded_pose(info, reference):
    """Relative pitch/yaw/roll degrees; never copy a driver's absolute pose."""
    return tuple(
        np.clip(np.asarray(info[key]) - np.asarray(reference[key]), -limit, limit)
        for key, limit in (("pitch", 10.0), ("yaw", 15.0), ("roll", 10.0))
    )
