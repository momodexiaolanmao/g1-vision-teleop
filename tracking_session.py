"""Explicit start, pause and input-loss handling shared by the live control loop."""
from dataclasses import dataclass
import math
import numpy as np


class CalibrationGate:
    """Give the operator time to leave the computer, then require a quiet pose."""
    def __init__(self, delay=8., stable_seconds=1.5):
        self.delay=delay
        self.stable_seconds=stable_seconds
        self.begin(0.)

    def begin(self, now):
        self.not_before=now+self.delay
        self.anchor=None
        self.stable_since=None

    def ready(self, observation, valid, now):
        points=getattr(observation,'landmarks',None)
        if now<self.not_before or not valid or not points or len(points)!=33:
            self.anchor=self.stable_since=None
            return False
        indices=(11,12,15,16,23,24,27,28,31,32)
        if not all(min(points[i].visibility or 0,points[i].presence or 0)>=.6 for i in indices):
            self.anchor=self.stable_since=None
            return False
        if not all(points[w].y>points[s].y+.04 for s,w in ((11,15),(12,16))):
            self.anchor=self.stable_since=None
            return False
        frame=getattr(observation,'frame',None)
        aspect=frame.shape[1]/frame.shape[0] if frame is not None else 16/9
        signature=np.array([[points[i].x*aspect,points[i].y] for i in indices])
        if not np.all(np.isfinite(signature)):
            self.anchor=self.stable_since=None
            return False
        if self.anchor is None or np.max(np.linalg.norm(signature-self.anchor,axis=1))>.014:
            self.anchor=signature
            self.stable_since=now
            return False
        return now-self.stable_since>=self.stable_seconds

    def display(self, now):
        if now<self.not_before:
            return f'GO TO CAMERA: {math.ceil(self.not_before-now)}s - full body, arms relaxed'
        if self.stable_since is None:
            return 'WAITING FOR FULL BODY - feet down, arms relaxed, stand still'
        return f'STAND STILL: {max(0.,self.stable_seconds-(now-self.stable_since)):.1f}s to calibrate'


@dataclass
class TrackingSession:
    enabled: bool = False
    ready: bool = False
    lost_latched: bool = False
    start_pending: bool = False
    invalid_since: float | None = None
    loss_timeout: float = 0.75

    def request_start(self):
        # Remember C even when no usable pose is available at that instant.
        self.start_pending = True
        self.enabled = False
        self.invalid_since = None

    def toggle_pause(self, valid):
        if self.enabled or self.start_pending:
            self.enabled = False
            self.start_pending = False
        elif self.ready and valid and not self.lost_latched:
            self.enabled = True
            self.invalid_since = None

    def advance(self, valid, now, stream_fresh=True, start_ready=True):
        """Return (restart, can_track). Missing input always holds immediately."""
        restart = False
        if self.start_pending and valid and start_ready:
            self.start_pending = False
            self.enabled = self.ready = True
            self.lost_latched = False
            self.invalid_since = None
            restart = True
        if not self.enabled:
            return restart, False
        if valid:
            self.invalid_since = None
            return restart, True
        if stream_fresh:
            # Normal occlusion in a live video keeps the user's start request.
            # Resume only when a reliable mapped joint is visible again.
            self.invalid_since = None
            return restart, False
        if self.invalid_since is None:
            self.invalid_since = now
        if now - self.invalid_since >= self.loss_timeout:
            self.enabled = False
            self.lost_latched = True
        return restart, False

    def status(self, valid, mapping=False, depth=False):
        if self.start_pending:
            return "START REQUESTED: waiting for a visible mapped joint"
        if self.lost_latched:
            return "CAMERA DATA STALE: fresh video must recover, then press C"
        if self.enabled and not valid:
            return "HOLDING: waiting for fresh visible joints"
        if self.enabled:
            if mapping:
                return "TRACKING (custom joint mapping)"
            return "TRACKING (estimated depth)" if depth else "TRACKING (image XY, fixed depth)"
        return "PAUSED" if self.ready else "Click this window and press C to start"
