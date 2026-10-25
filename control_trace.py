"""Optional numeric input/physics recording for deterministic local diagnosis.

No frames are saved. ControlTrace records joint goals and physical feedback;
PoseTrace optionally records numeric landmarks during an explicit timed sample.
"""
import json
from pathlib import Path


class ControlTrace:
    def __init__(self, path, robot):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        self.file=path.open('w',encoding='utf-8')
        self.write({'type':'metadata','version':1,'model':robot.model_path,
                    'mapping':robot.config,'qpos':robot.data.qpos.tolist(),
                    'qvel':robot.data.qvel.tolist(),'time':robot.data.time})

    def write(self, value):
        self.file.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n')

    def sample(self, robot):
        balance=robot.balance
        locomotion=getattr(balance,'locomotion',balance)
        self.write({'type':'input','time':robot.data.time,
                    'goal':robot.goal.tolist(),'command':robot.command.tolist(),
                    'step_active':balance.intent_active,
                    'root_intent':balance.root_intent,
                    'origin':balance.origin.tolist(),'origin_yaw':balance.origin_yaw,
                    'calibrated':robot.retarget.calibrated,
                    'foot_intent':robot.footstep.status(),
                    'phase':balance.phase,'fallen':balance.fallen,
                    'velocity_command':balance.command.tolist(),
                    'intent_phase':getattr(locomotion,'intent_phase',None),
                    'step_direction':getattr(locomotion,'step_direction',None),
                    'qpos':robot.data.qpos.tolist(),'qvel':robot.data.qvel.tolist()})
        self.file.flush()

    def close(self):
        self.file.close()
class PoseTrace:
    """Optional local numeric observations for reproducing capture/filter gaps."""
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.file=self.path.open('w',encoding='utf-8')
        self.previous=None

    def sample(self,observation):
        if observation is None or observation.arrived==self.previous:
            return
        self.previous=observation.arrived
        image=getattr(observation,'landmarks',[]) or []
        world=getattr(observation,'world_landmarks',[]) or []
        record={'time':observation.arrived,'image_size':[observation.frame.shape[1],observation.frame.shape[0]],
                'subject_state':observation.subject_state,
                'image':[[p.x,p.y,p.z,p.visibility or 0,p.presence or 0] for p in image],
                'world':[[p.x,p.y,p.z] for p in world]}
        self.file.write(json.dumps(record)+'\n')
        self.file.flush()

    def close(self):
        self.file.close()
