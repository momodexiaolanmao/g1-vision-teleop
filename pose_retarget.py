"""Geometric, visibility-aware body channels from MediaPipe's 33 landmarks.

Angles are anatomical channels in degrees, independent of robot joint names.
Axial rotations and palm orientation are estimates from bend/hand planes, not
measurements of human joints. Singular planes are held rather than invented.
"""
import math
import numpy as np

LANDMARK_NAMES = ("nose left_eye_inner left_eye left_eye_outer right_eye_inner right_eye "
                  "right_eye_outer left_ear right_ear mouth_left mouth_right left_shoulder "
                  "right_shoulder left_elbow right_elbow left_wrist right_wrist left_pinky "
                  "right_pinky left_index right_index left_thumb right_thumb left_hip "
                  "right_hip left_knee right_knee left_ankle right_ankle left_heel right_heel "
                  "left_foot_index right_foot_index").split()
IDS = {name: i for i, name in enumerate(LANDMARK_NAMES)}
CHANNEL_LABELS = {"waist_yaw": "腰部扭转", "waist_roll": "腰部侧倾", "waist_pitch": "腰部前倾"}
for side, label in (("left", "左"), ("right", "右")):
    for suffix, title in (("hip_pitch", "髋前后摆"), ("hip_roll", "髋侧摆"),
                          ("hip_yaw", "大腿旋转"), ("knee_flex", "膝弯曲"),
                          ("ankle_pitch", "踝前后倾"), ("ankle_roll", "踝侧倾"),
                          ("shoulder_pitch", "肩前后抬臂"), ("shoulder_roll", "肩侧抬臂"),
                          ("shoulder_yaw", "上臂旋转"), ("elbow_flex", "肘弯曲"),
                          ("wrist_roll", "手腕旋转"), ("wrist_pitch", "手腕屈伸"),
                          ("wrist_yaw", "手腕侧摆")):
        CHANNEL_LABELS[f"{side}_{suffix}"] = label + title


def unit(vector):
    length = np.linalg.norm(vector)
    return None if not np.isfinite(length) or length < 1e-6 else vector / length


def basis(left, up):
    """Columns are robot-forward, anatomical-left, and up in camera space."""
    y = unit(left)
    x = None if y is None else unit(np.cross(y, up))
    if x is None:
        return None
    return np.column_stack((x, y, np.cross(x, y)))


def rotate(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def xyz_angles(matrix):
    # Intrinsic X/Y/Z, matching the G1 wrist chain.
    pitch = math.asin(float(np.clip(matrix[0, 2], -1, 1)))
    if abs(math.cos(pitch)) < 0.05:
        return None
    return np.degrees([math.atan2(-matrix[1, 2], matrix[2, 2]), pitch,
                       math.atan2(-matrix[0, 1], matrix[0, 0])])


class BodyRetarget:
    def __init__(self, confidence=0.6, quality_config=None):
        self.confidence = confidence
        self.quality_config = quality_config or {}
        self.reset()

    def reset(self):
        self.calibrated = False
        self.reference_basis = None
        self.reference_hip = None
        self.reference_scale = None
        self.wrist_neutral = {}
        self.last_torso = None
        self.last_angles = {}
        self.cache_observation = None
        self.cache_result = None
        self.neutral_offsets = {}
        self.angle_history = {}
        self.image_history = {}
        self.pending_angles = {}
        self.pending_image_motion = {}

    def _load(self, observation):
        self.observation = observation
        self.image = getattr(observation, "landmarks", None)
        self.world = getattr(observation, "world_landmarks", None)
        self.points = {}
        self.rejected_bones=[]
        if self.image is None or self.world is None or len(self.image) != 33 or len(self.world) != 33:
            return
        for name, index in IDS.items():
            point, quality = self.world[index], self.image[index]
            values = np.array([point.x, point.y, point.z], dtype=float)
            scores = [quality.visibility or 0, quality.presence or 0]
            in_frame = (np.isfinite(quality.x) and np.isfinite(quality.y)
                        and 0 <= quality.x <= 1 and 0 <= quality.y <= 1)
            if in_frame and np.all(np.isfinite(values)) and all(np.isfinite(q) and q >= self.confidence for q in scores):
                self.points[name] = values
        self.rejected_bones = []
        if self.quality_config.get("enabled", False):
            for side in ("left", "right"):
                for a,b,c in (("shoulder","elbow","wrist"),("hip","knee","ankle")):
                    first,middle,last=(f"{side}_{n}" for n in (a,b,c))
                    if self._has(first,middle,last):
                        upper=np.linalg.norm(self.points[middle]-self.points[first])
                        lower=np.linalg.norm(self.points[last]-self.points[middle])
                        low,high=(.10,.43) if a=="shoulder" else (.18,.58)
                        if not (low<=lower<=high and .50<=lower/max(upper,1e-6)<=1.45):
                            self.points.pop(last,None)
                            if last.endswith("wrist"):
                                for finger in ("index","pinky","thumb"):
                                    self.points.pop(f"{side}_{finger}",None)
                            self.rejected_bones.append(f"{side}_{b}_{c}")

    def _has(self, *names):
        return all(name in self.points for name in names)

    def _frames(self):
        p = self.points
        pelvis, torso = None, None
        if self._has("left_hip", "right_hip"):
            pelvis = basis(p["left_hip"] - p["right_hip"], np.array([0., -1., 0.]))
        if self._has("left_shoulder", "right_shoulder"):
            up = np.array([0., -1., 0.]) if self.last_torso is None else self.last_torso[:, 2]
            if self._has("left_hip", "right_hip"):
                up = (p["left_shoulder"] + p["right_shoulder"] - p["left_hip"] - p["right_hip"]) / 2
            torso = basis(p["left_shoulder"] - p["right_shoulder"], up)
            if torso is not None:
                self.last_torso = torso.copy()
        return pelvis, torso

    def calibration_ready(self, observation):
        self._load(observation)
        return self._has("left_hip", "right_hip", "left_shoulder", "right_shoulder") and self._frames()[0] is not None

    def _image_scale(self):
        image = self.image
        aspect = self.observation.frame.shape[1] / self.observation.frame.shape[0]
        shoulder = np.array([(image[11].x + image[12].x) * aspect / 2,
                             (image[11].y + image[12].y) / 2])
        hip = np.array([(image[23].x + image[24].x) * aspect / 2,
                       (image[23].y + image[24].y) / 2])
        return hip, float(np.linalg.norm(shoulder - hip))

    def calibrate(self, observation):
        if not self.calibration_ready(observation):
            return False
        pelvis, _ = self._frames()
        hip, scale = self._image_scale()
        if not np.all(np.isfinite(hip)) or scale < 0.025:
            return False
        self.reference_basis, self.reference_hip, self.reference_scale = pelvis.copy(), hip, scale
        self.calibrated = True
        self.cache_observation = None
        self.neutral_offsets = {}
        self.angle_history = {}
        self.pending_angles = {}
        self.image_history = {}
        self.pending_image_motion = {}
        self.wrist_neutral = {}
        # A relaxed hand at C defines its neutral orientation; each hand can
        # initialise later if hidden during calibration.
        channels, _ = self.evaluate(observation)
        if self.quality_config.get("neutral_correction", False):
            for name, channel in channels.items():
                # Flexion retains its anatomical meaning. Correct small rest
                # biases, never zero a raised arm or a deeply bent knee at C.
                if (channel["valid"] and abs(channel["angle"]) <= 25
                        and not name.endswith("flex") and "wrist" not in name):
                    self.neutral_offsets[name] = channel["angle"]
        self.angle_history = {}
        self.cache_observation = None
        return True

    def _image_signature(self, name):
        return self._image_evidence(name)[0]

    def _image_evidence(self, name):
        aspect=self.observation.frame.shape[1]/self.observation.frame.shape[0]
        arm=any(n in name for n in ('shoulder','elbow','wrist'))
        if arm:
            # Visible arms remain controllable when the hips leave the frame.
            # Use the same shoulder anchor with or without hips so losing them
            # cannot disable stationary-image rejection or change its shape.
            names=['left_shoulder','right_shoulder']
            if not self._has(*names):
                return None,None
            shoulders=np.array([[self.image[IDS[n]].x*aspect,self.image[IDS[n]].y] for n in names])
            origin=np.mean(shoulders,axis=0)
            scale=float(np.linalg.norm(shoulders[0]-shoulders[1]))
        else:
            names=['left_shoulder','right_shoulder','left_hip','right_hip']
            if not self._has(*names):
                return None,None
            origin,scale=self._image_scale()
        if scale<.025:
            return None,None
        if not name.startswith('waist'):
            side=name.split('_')[0]
            limb=('shoulder','elbow','wrist') if arm else ('hip','knee','ankle')
            names += [side+'_'+n for n in limb]
            if 'wrist' in name:
                names += [side+'_'+n for n in ('index','pinky')]
            elif 'ankle' in name:
                names += [side+'_'+n for n in ('heel','foot_index')]
        if not self._has(*names):
            return None,None
        coordinates=np.array([[self.image[IDS[n]].x*aspect,self.image[IDS[n]].y] for n in names])
        # Both coordinates use image-height units after aspect correction.
        # This converts distances in the normalized signature back to pixels.
        return (coordinates-origin)/scale,scale*self.observation.frame.shape[0]

    def _quality_angle(self, name, value):
        """Reject isolated reconstruction flips, then adapt smoothing to speed."""
        value -= self.neutral_offsets.get(name, 0.)
        if not self.quality_config.get("enabled", False):
            return value, None
        stamp = float(getattr(self.observation, "arrived", 0.))
        signature,pixels_per_signature=self._image_evidence(name)
        if signature is None:
            self.pending_image_motion.pop(name,None)
            self.pending_angles.pop(name,None)
            return None,"reliable image evidence missing; HOLD"
        previous = self.angle_history.get(name)
        if previous is not None:
            old_stamp, old_raw, filtered, old_velocity = previous
            dt = stamp - old_stamp
            if dt<=0:
                return filtered,None
            if dt>0:
                # Missing people or rejected bones can leave a long time gap.
                # Elapsed time alone cannot authorize a new reconstructed pose
                # or replace its last accepted image reference.
                sample_dt=min(dt,.05)
                if dt>=.3:
                    old_velocity=0.
                    self.pending_angles.pop(name,None)
                delta = value - old_raw
                saved=self.image_history.get(name)
                image_supported=True
                if signature is not None and saved is not None:
                    threshold=max(.02,float(self.quality_config.get('min_image_motion_px',10))/pixels_per_signature)
                    if np.max(np.linalg.norm(signature-saved,axis=1))<threshold:
                        self.pending_image_motion.pop(name,None)
                        image_supported=False
                    else:
                        prior_stamp,count=self.pending_image_motion.get(name,(-math.inf,0))
                        if stamp>prior_stamp:
                            count=count+1 if stamp-prior_stamp<.3 else 1
                            self.pending_image_motion[name]=(stamp,count)
                        image_supported=count>=int(self.quality_config.get('image_motion_confirmation_frames',3))
                else:
                    self.pending_image_motion.pop(name,None)
                limit = 12 + float(self.quality_config.get("max_angular_speed_deg_s", 650)) * sample_dt
                if abs(delta) > limit:
                    pending, count = self.pending_angles.get(name, (value, 0))
                    count = count + 1 if abs(pending - value) < 15 else 1
                    self.pending_angles[name] = value, count
                    # A persistent new pose can recover after 3 observations;
                    # a one-frame depth spike cannot alter the motor reference.
                    if count < 3 and image_supported:
                        return None, "reconstruction jump; HOLD"
                else:
                    self.pending_angles.pop(name,None)
                if not image_supported:
                    # Depth reconstruction may drift while the observed
                    # silhouette stays still. Keep the last supported angle;
                    # release requires sustained image displacement from the
                    # last accepted reference, so slow motion can accumulate.
                    value=old_raw
                    delta=0.
                elif signature is not None:
                    self.image_history[name]=signature.copy()
                    self.pending_image_motion.pop(name,None)
                velocity = delta / sample_dt
                va = 1 - math.exp(-2 * math.pi * 2. * sample_dt)
                velocity = old_velocity + va * (velocity - old_velocity)
                cutoff = float(self.quality_config.get("min_cutoff_hz", 3.0)) + .12 * abs(velocity)
                alpha = 1 - math.exp(-2 * math.pi * cutoff * sample_dt)
                value_filtered = filtered + alpha * (value - filtered)
                self.angle_history[name] = stamp, value, value_filtered, velocity
                if image_supported:
                    self.pending_angles.pop(name, None)
                return value_filtered, None
        self.angle_history[name] = stamp, value, value, 0.
        if signature is not None:
            self.image_history[name]=signature.copy()
        self.pending_angles.pop(name, None)
        self.pending_image_motion.pop(name,None)
        return value, None

    def evaluate(self, observation):
        if observation is self.cache_observation and self.cache_result is not None:
            return self.cache_result
        self._load(observation)
        channels = {name: {"valid": False, "reason": "landmarks hidden or degenerate", "missing": []}
                    for name in CHANNEL_LABELS}
        root = {"valid": False, "reason": "press C with shoulders and hips visible"}
        if not self.points:
            self.pending_image_motion.clear()
            self.pending_angles.clear()
            result = channels, root
            self.cache_observation, self.cache_result = observation, result
            return result
        p = self.points
        pelvis, torso = self._frames()
        observed_angles=set()

        def put(channel, value, required=()):
            # Monocular flips can yield a 180-degree waist/hip twist despite
            # high landmark visibility. Such an anatomical rotation is an
            # ambiguous reconstruction, not a useful dance command.
            cap = (110 if channel in ("waist_yaw", "waist_pitch") else
                   100 if channel.endswith("hip_yaw") or channel.endswith("wrist_yaw")
                        or self.quality_config.get("enabled",False) and channel.endswith("shoulder_yaw") else
                   150 if channel.endswith("wrist_roll") else None)
            if cap is not None and value is not None and abs(value) > cap:
                channels[channel]["reason"] = "ambiguous rotation; HOLD"
                return
            if value is not None and np.isfinite(value):
                raw_value = float(value)
                observed_angles.add(channel)
                filtered, reason = self._quality_angle(channel, raw_value)
                if filtered is None:
                    channels[channel]["reason"] = reason
                    return
                channels[channel] = {"valid": True, "angle": float(filtered),
                                     "raw_angle_deg": raw_value, "reason": "tracking", "missing": []}
                self.last_angles[channel] = raw_value
            else:
                channels[channel]["missing"] = [n for n in required if n not in p]

        if pelvis is not None and torso is not None and self._has("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
            relative = pelvis.T @ torso
            roll = math.asin(float(np.clip(relative[2, 1], -1, 1)))
            put("waist_roll", math.degrees(roll))
            put("waist_yaw", math.degrees(math.atan2(-relative[0, 1], relative[1, 1])))
            put("waist_pitch", math.degrees(math.atan2(-relative[2, 0], relative[2, 2])))
        for side in ("left", "right"):
            for limb, frame, first, middle, last in (("shoulder", torso, "shoulder", "elbow", "wrist"),
                                                     ("hip", pelvis, "hip", "knee", "ankle")):
                a, b, c = (f"{side}_{n}" for n in (first, middle, last))
                upper = unit(p[b] - p[a]) if self._has(a, b) else None
                lower = unit(p[c] - p[b]) if self._has(b, c) else None
                flex_name = f"{side}_{'elbow' if limb == 'shoulder' else 'knee'}_flex"
                flex = None
                if upper is not None and lower is not None:
                    flex = math.acos(float(np.clip(np.dot(upper, lower), -1, 1)))
                    put(flex_name, math.degrees(flex))
                else:
                    put(flex_name, None, (a, b, c))
                chain = None
                if frame is not None and upper is not None:
                    v = frame.T @ upper
                    roll = math.asin(float(np.clip(v[1], -1, 1)))
                    pitch = math.atan2(-v[0], -v[2])
                    if limb == "shoulder" and pitch > math.radians(170):
                        pitch -= 2 * math.pi
                    # Pitch cannot be inferred when a limb is exactly lateral.
                    if math.hypot(v[0], v[2]) > 0.08:
                        put(f"{side}_{limb}_pitch", math.degrees(pitch))
                    put(f"{side}_{limb}_roll", math.degrees(roll))
                    pr = rotate("y", pitch) @ rotate("x", roll)
                    yaw = None
                    bend_threshold=.35 if self.quality_config.get("enabled",False) else .15
                    if lower is not None and flex is not None and abs(math.sin(flex)) > bend_threshold:
                        bend = frame.T @ (lower - upper * np.dot(upper, lower))
                        local = pr.T @ bend
                        sign = 1 if limb == "shoulder" else -1
                        yaw = math.atan2(sign * local[1], sign * local[0])
                    elif limb == "hip" and self._has(f"{side}_heel", f"{side}_foot_index"):
                        foot = frame.T @ (p[f"{side}_foot_index"] - p[f"{side}_heel"])
                        if np.linalg.norm(foot[:2]) > 0.025:
                            yaw = math.atan2(foot[1], foot[0])
                    put(f"{side}_{limb}_yaw", None if yaw is None else math.degrees(yaw), (a, b, c))
                    prior = math.radians(self.last_angles.get(f"{side}_{limb}_yaw", 0))
                    chain_yaw=prior if yaw is None else yaw
                    if self.quality_config.get('enabled',False):
                        if not channels[f'{side}_{limb}_yaw']['valid']:
                            chain_yaw=math.radians(self.last_angles.get(f'{side}_{limb}_yaw',0))
                        if not all(channels[f'{side}_{limb}_{axis}']['valid'] for axis in ('pitch','roll')):
                            continue
                    chain = frame @ pr @ rotate("z",chain_yaw)
                if limb == "shoulder" and upper is not None and lower is not None and chain is not None:
                    self._wrists(side, upper, lower, chain, channels, put)
                if limb == "hip" and lower is not None and chain is not None and flex is not None:
                    self._ankles(side, chain @ rotate("y", flex), put)
        if pelvis is not None and self.calibrated and self._has("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            relative = self.reference_basis.T @ pelvis
            hip, scale = self._image_scale()
            yaw = math.atan2(relative[1, 0], relative[0, 0])
            roll = math.atan2(relative[2, 1], relative[2, 2])
            if np.all(np.isfinite(hip)) and scale > 0.025:
                meters_per_image = 0.42 / self.reference_scale
                left_sign = np.sign(self.reference_basis[0, 1]) or 1
                root = {"valid": True, "reason": "tracking", "yaw_rad": yaw, "roll_rad": roll,
                        "lateral_m": float((hip[0] - self.reference_hip[0]) * meters_per_image * left_sign),
                        "vertical_m": float((self.reference_hip[1] - hip[1]) * meters_per_image),
                        "scale_ratio": self.reference_scale / scale,
                        "depth_reliable": bool(abs(math.cos(yaw)) > 0.7 and torso is not None
                            and np.dot(torso[:, 2], np.array([0., -1., 0.])) > 0.87)}
        # Occlusion breaks the sequence of supported fresh image samples.
        for name in set(self.pending_image_motion)-observed_angles:
            self.pending_image_motion.pop(name,None)
        for name in set(self.pending_angles)-observed_angles:
            self.pending_angles.pop(name,None)
        result = channels, root
        self.cache_observation, self.cache_result = observation, result
        return result

    def _wrists(self, side, upper, lower, chain, channels, put):
        p = self.points
        required = [f"{side}_{n}" for n in ("wrist", "index", "pinky")]
        if not self._has(*required):
            for axis in ("roll", "pitch", "yaw"):
                put(f"{side}_wrist_{axis}", None, required)
            return
        wrist, index, pinky = (p[n] for n in required)
        hand_x = unit((index + pinky) / 2 - wrist)
        side_vector = index - pinky
        hand_y = None if hand_x is None else unit(side_vector - hand_x * np.dot(side_vector, hand_x))
        perpendicular = None if hand_x is None else side_vector - hand_x * np.dot(side_vector, hand_x)
        if (hand_y is None or np.linalg.norm(side_vector) < 0.015
                or np.linalg.norm(perpendicular) / np.linalg.norm(side_vector) < 0.3):
            return
        if self.quality_config.get("enabled",False):
            height,width=self.observation.frame.shape[:2]
            a,b=(self.image[IDS[f"{side}_{n}"]] for n in ("index","pinky"))
            if math.hypot((a.x-b.x)*width,(a.y-b.y)*height)<5:
                for axis in ("roll","pitch","yaw"):
                    channels[f"{side}_wrist_{axis}"]["reason"]="palm too small or edge-on; HOLD"
                return
        hand = np.column_stack((hand_x, hand_y, np.cross(hand_x, hand_y)))
        # The G1's elbow zero forearm points forward; human straight elbow
        # corresponds approximately to robot elbow +90 degrees.
        forearm_y = chain[:, 1] - lower * np.dot(chain[:, 1], lower)
        forearm_y = unit(forearm_y)
        if forearm_y is None:
            return
        forearm = np.column_stack((lower, forearm_y, np.cross(lower, forearm_y)))
        relative = forearm.T @ hand
        if side not in self.wrist_neutral:
            self.wrist_neutral[side] = relative.copy()
        angles = xyz_angles(relative @ self.wrist_neutral[side].T)
        if angles is not None:
            for axis, value in zip(("roll", "pitch", "yaw"), angles):
                put(f"{side}_wrist_{axis}", value)

    def _ankles(self, side, shank, put):
        p = self.points
        required = [f"{side}_{n}" for n in ("ankle", "heel", "foot_index")]
        if not self._has(*required):
            for axis in ("pitch", "roll"):
                put(f"{side}_ankle_{axis}", None, required)
            return
        ankle, heel, toe = (p[n] for n in required)
        x = unit(toe - heel)
        up = ankle - (heel + toe) / 2
        z = None if x is None else unit(up - x * np.dot(x, up))
        if z is None:
            return
        foot = np.column_stack((x, np.cross(z, x), z))
        relative = shank.T @ foot
        put(f"{side}_ankle_pitch", math.degrees(math.atan2(-relative[2, 0], relative[0, 0])))
        put(f"{side}_ankle_roll", math.degrees(math.atan2(-relative[1, 2], relative[1, 1])))
