"""Conservative spatial torso tracking; detector order is not person identity.

This is not biometric identification. It locks a torso location/scale, refuses
large jumps to another detection, and requires an explicit reset to reacquire.
"""
import numpy as np


class SubjectTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.center=None
        self.scale=None
        self.width=None
        self.locked=False
        self.state='WAITING FOR FOREGROUND PERSON'

    def lock(self):
        if self.center is not None:
            self.locked=True
            self.state='FOREGROUND PERSON LOCKED'

    def lock_to(self, center, scale, width=None):
        self.center=np.asarray(center).copy()
        self.scale=float(scale)
        self.width=None if width is None else float(width)
        self.lock()

    def _candidate(self, points, aspect):
        if len(points)!=33:
            return None
        def visible(index):
            p=points[index]
            return (np.isfinite(p.x) and np.isfinite(p.y) and 0<=p.x<=1 and 0<=p.y<=1
                    and min(p.visibility or 0,p.presence or 0)>=.55)
        if not all(visible(i) for i in (11,12)):
            return None
        xy=lambda index:np.array([points[index].x*aspect,points[index].y])
        center=(xy(11)+xy(12))/2
        width=np.linalg.norm(xy(11)-xy(12))
        if all(visible(i) for i in (23,24)):
            scale=np.linalg.norm(center-(xy(23)+xy(24))/2)
        else:
            scale=self.scale if self.scale is not None else width*1.4
        if not .025<scale<1.2 or width<.015:
            return None
        return center,float(scale),float(width)

    def select(self, poses, aspect):
        candidates=[(i,c) for i,p in enumerate(poses) if (c:=self._candidate(p,aspect)) is not None]
        if not self.locked:
            if not candidates:
                return None
            def score(item):
                center,scale,width=item[1]
                distance=abs(center[0]-aspect/2)
                return scale*np.sqrt(width)/(1+distance)
            index,(center,scale,width)=max(candidates,key=score)
        else:
            compatible=[(i,c) for i,c in candidates if
                        np.linalg.norm(c[0]-self.center)<max(.06,.45*self.scale)
                        and .5<c[1]/self.scale<1.8
                        and (self.width is None or .5<c[2]/self.width<1.8)]
            if not compatible:
                self.state='SELECTED PERSON LOST - HOLD'
                return None
            index,(center,scale,width)=min(compatible,key=lambda item:
                np.linalg.norm(item[1][0]-self.center)+.15*abs(item[1][1]-self.scale))
        self.center=center.copy()
        self.scale=scale
        self.width=width
        self.state='FOREGROUND PERSON LOCKED' if self.locked else 'FOREGROUND READY - press C to lock'
        return index
