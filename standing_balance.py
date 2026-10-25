"""Torque-only whole-body inverse dynamics with fixed double-support tasks.

The QP's contact forces are predictions, never applied to the free base.
Only bounded joint motor commands enter MuJoCo; its ground contacts support
the actual robot. Foot tasks and COM control take priority over human posture.
"""
import math

import mujoco
import numpy as np


def orientation_error(current, target):
    return 0.5 * sum(np.cross(current[:, i], target[:, i]) for i in range(3))


class StandingBalance:
    def __init__(self, model, data, config=None):
        self.model, self.data, self.config = model, data, config or {}
        self.backend = self.config.get("solver", "numpy")
        if self.backend == "osqp":
            import osqp
            from scipy import sparse
            self.osqp, self.sparse = osqp, sparse
        elif self.backend != "numpy":
            raise ValueError("Standing balance solver must be numpy or osqp")
        if model.nv != 35 or model.nu != 29 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Standing balance requires the free-base 29-joint G1 model")
        if model.neq or model.nmocap or np.any(model.body_gravcomp):
            raise ValueError("Standing balance must use gravity, no welds, mocap or gravity compensation")
        if not np.allclose(model.opt.gravity, [0, 0, -9.81]):
            raise ValueError("Standing balance requires normal Earth gravity")
        self.period = float(self.config.get("control_period_s", 0.006))
        self.mu = float(self.config.get("friction", 0.65))
        self.max_crouch = float(self.config.get("max_crouch_m", 0.12))
        self.max_shift = float(self.config.get("max_com_shift_m", 0.02))
        speed = float(self.config.get("upper_body_speed_deg_s", 90))
        if not (0.002 <= self.period <= 0.02 and 0 < self.mu <= 0.8
                and 0 <= self.max_crouch <= 0.16 and 0 <= self.max_shift <= 0.025):
            raise ValueError("Invalid standing balance period, friction or support motion limits")
        if not 0 < speed <= 180:
            raise ValueError("Standing reference speed must be 0..180 degrees/second")
        for axis, default, ceiling in (("yaw", 35, 60), ("roll", 12, 20), ("pitch", 18, 25)):
            if not 0 <= float(self.config.get(f"waist_{axis}_limit_deg", default)) <= ceiling:
                raise ValueError(f"Standing waist {axis} limit must be 0..{ceiling} degrees")
        self.pelvis = model.body("pelvis").id
        self.feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
        self.foot_geoms = [i for i in range(model.ngeom)
                           if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
                           and model.geom_bodyid[i] in self.feet]
        if len(self.foot_geoms) != 8:
            raise ValueError("Expected the official eight G1 foot contact spheres")
        self.joints = [int(model.actuator_trnid[a, 0]) for a in range(model.nu)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in self.joints])
        self.names = [model.joint(j).name for j in self.joints]
        self.index = {name: i for i, name in enumerate(self.names)}
        if sorted(self.vadr.tolist()) != list(range(6, 35)) or not np.allclose(model.actuator_gear[:, 0], 1):
            raise ValueError("Expected one direct torque motor per G1 hinge")
        self.nv, self.nf = model.nv, 3 * len(self.foot_geoms)
        self.nx = self.nv + self.nf
        self.mass = np.empty((self.nv, self.nv))
        self.probe = mujoco.MjData(model)
        self.solver = None
        self.reset()

    def reset(self):
        d, m = self.data, self.model
        self.home = d.qpos[self.qadr].copy()
        self.reference = self.home.copy()
        self.foot_positions = [d.xpos[b].copy() for b in self.feet]
        self.foot_rotations = [d.xmat[b].reshape(3, 3).copy() for b in self.feet]
        self.base_rotation = d.xmat[self.pelvis].reshape(3, 3).copy()
        centers = np.array([d.geom_xpos[g] for g in self.foot_geoms])
        self.home_com = d.subtree_com[self.pelvis].copy()
        self.home_com[:2] = np.mean(centers[:, :2], axis=0)
        self.com_goal = self.home_com.copy()
        self.last_update = -math.inf
        self.solves, self.failures = 0, 0
        self.last_solver_status = "initializing"
        self.last_torque = np.zeros(m.nu)
        self.max_torque_ratio = 0.
        self.fallen = False
        self.limited_joints = []
        self.solve_time_ms = 0.
        self.solver = None

    def set_reference(self, mapped_angles, entries):
        reference = self.home.copy()
        limited = []
        for value, entry in zip(mapped_angles, entries):
            name = entry["rule"]["robot_joint"]
            index = self.index[name]
            bounded = float(value)
            if name.startswith("waist_"):
                axis = name.split("_")[1]
                limit = math.radians(float(self.config.get(f"waist_{axis}_limit_deg",
                                                          {"yaw": 35, "roll": 12, "pitch": 18}[axis])))
                bounded = float(np.clip(value, -limit, limit))
            if abs(bounded - value) > 1e-6:
                limited.append(name)
            reference[index] = bounded
        self.reference = reference
        self.limited_joints = limited
        knees = [self.index[s + "_knee_joint"] for s in ("left", "right")]
        # The initial bent-knee support posture is not a human squat command.
        crouch = self.max_crouch * float(np.clip((np.mean(reference[knees]) - np.mean(self.home[knees])) / 1.1, 0, 1))
        self.com_goal = self.home_com.copy()
        self.com_goal[2] -= crouch
        rolls = [self.index[s + "_hip_roll_joint"] for s in ("left", "right")]
        self.com_goal[1] += self.max_shift * float(np.clip(np.mean(reference[rolls]) / 0.25, -1, 1))

    def _point_jacobian(self, body, point):
        m, d = self.model, self.data
        jp, jr = np.zeros((3, self.nv)), np.zeros((3, self.nv))
        dp, dr = np.zeros_like(jp), np.zeros_like(jr)
        mujoco.mj_jac(m, d, jp, jr, point, body)
        mujoco.mj_jacDot(m, d, dp, dr, point, body)
        return jp, jr, dp @ d.qvel, dr @ d.qvel

    def _solve(self):
        import time
        started = time.perf_counter()
        m, d, nv, nx = self.model, self.data, self.nv, self.nx
        mujoco.mj_fullM(m, d, self.mass)
        bias = d.qfrc_bias - d.qfrc_passive
        contact_jacobian = np.vstack([self._point_jacobian(int(m.geom_bodyid[g]), d.geom_xpos[g])[0]
                                      for g in self.foot_geoms])
        dynamics = np.hstack([self.mass, -contact_jacobian.T])
        equality = [dynamics[:6]]
        targets = [-bias[:6]]
        for body, pos, rotation in zip(self.feet, self.foot_positions, self.foot_rotations):
            jp, jr, bp, br = self._point_jacobian(body, d.xpos[body])
            jac = np.vstack([jp, jr])
            error = np.r_[pos - d.xpos[body],
                          orientation_error(d.xmat[body].reshape(3, 3), rotation)]
            acc = 100 * error - 20 * (jac @ d.qvel) - np.r_[bp, br]
            equality.append(np.hstack([jac, np.zeros((6, self.nf))]))
            targets.append(acc)
        tasks, wanted = [], []

        def task(jac, acceleration, weight):
            tasks.append(np.hstack([jac, np.zeros((len(jac), self.nf))]) * weight)
            wanted.append(acceleration * weight)

        jc = np.zeros((3, nv))
        mujoco.mj_jacSubtreeCom(m, d, jc, self.pelvis)
        self.probe.qpos[:] = d.qpos
        self.probe.qvel[:] = d.qvel
        eps = 1e-5
        mujoco.mj_integratePos(m, self.probe.qpos, d.qvel, eps)
        mujoco.mj_forward(m, self.probe)
        jc_next = np.zeros_like(jc)
        mujoco.mj_jacSubtreeCom(m, self.probe, jc_next, self.pelvis)
        com_bias = (jc_next - jc) @ d.qvel / eps
        task(jc, 100 * (self.com_goal - d.subtree_com[self.pelvis]) - 20 * (jc @ d.qvel) - com_bias, 20.)
        _, jb, _, bb = self._point_jacobian(self.pelvis, d.xpos[self.pelvis])
        task(jb, 100 * orientation_error(d.xmat[self.pelvis].reshape(3, 3), self.base_rotation)
             - 20 * (jb @ d.qvel) - bb, 6.)
        posture = np.eye(nv)[self.vadr]
        acceleration = 70 * (self.reference - d.qpos[self.qadr]) - 17 * d.qvel[self.vadr]
        # Legs are support joints: human leg angles are a low-priority posture
        # preference. Foot contact and balance determine their actual motion.
        weights = np.array([0.08 if any(t in n for t in ("hip", "knee", "ankle")) else 1.0
                            for n in self.names])
        task(posture * weights[:, None], acceleration * weights, 2.)
        task(np.eye(nv), -2 * d.qvel, 0.03)
        objectives = np.vstack(tasks)
        desired = np.concatenate(wanted)
        p = objectives.T @ objectives + np.diag(np.r_[np.full(nv, 1e-5), np.full(self.nf, 1e-5)])
        q = -objectives.T @ desired
        bounds = np.eye(nx)[:nv]
        lo, hi = np.full(nv, -150.), np.full(nv, 150.)
        # Short-horizon acceleration/velocity limits anticipate joint stops.
        horizon = 0.08
        limits = m.jnt_range[self.joints]
        lo[self.vadr] = np.maximum(lo[self.vadr], 2 * (limits[:, 0] + 0.015 - d.qpos[self.qadr]
                                                     - horizon * d.qvel[self.vadr]) / horizon**2)
        hi[self.vadr] = np.minimum(hi[self.vadr], 2 * (limits[:, 1] - 0.015 - d.qpos[self.qadr]
                                                     - horizon * d.qvel[self.vadr]) / horizon**2)
        lo[self.vadr] = np.maximum(lo[self.vadr], (-4 - d.qvel[self.vadr]) / horizon)
        hi[self.vadr] = np.minimum(hi[self.vadr], (4 - d.qvel[self.vadr]) / horizon)
        friction, fl, fu = [], [], []
        for i in range(len(self.foot_geoms)):
            start = nv + 3 * i
            for axis in (0, 1):
                for sign in (-1., 1.):
                    row = np.zeros(nx)
                    # Inscribed square: simultaneous x/y forces must also fit
                    # inside the physical circular Coulomb friction cone.
                    row[start + axis], row[start + 2] = sign, -self.mu / math.sqrt(2)
                    friction.append(row)
                    fl.append(-np.inf)
                    fu.append(0.)
            row = np.zeros(nx)
            row[start + 2] = 1.
            friction.append(row)
            fl.append(0.)
            fu.append(float(m.body_mass.sum() * 9.81))
        eq, target = np.vstack(equality), np.concatenate(targets)
        torque_rows = dynamics[self.vadr]
        a = np.vstack([eq, torque_rows, bounds, friction])
        lower = np.r_[target, m.actuator_ctrlrange[:, 0] - bias[self.vadr], lo, fl]
        upper = np.r_[target, m.actuator_ctrlrange[:, 1] - bias[self.vadr], hi, fu]
        if np.any(lower > upper):
            self.last_solver_status = "joint bounds infeasible"
            return None
        if self.backend == "numpy":
            from dense_qp import DenseQP
            if self.solver is None:
                self.solver = DenseQP()
            result = self.solver.solve(p, q, a, lower, upper)
        elif self.solver is None:
            # Explicit dense structural patterns allow efficient numerical
            # updates without dropping values which happen to be zero now.
            self.p_rows, self.p_cols = np.triu_indices(nx)
            self.a_rows = np.tile(np.arange(len(a)), nx)
            self.a_cols = np.repeat(np.arange(nx), len(a))
            pm = self.sparse.csc_matrix((p[self.p_rows, self.p_cols], (self.p_rows, self.p_cols)), shape=p.shape)
            am = self.sparse.csc_matrix((a[self.a_rows, self.a_cols], (self.a_rows, self.a_cols)), shape=a.shape)
            self.p_rows = pm.indices.copy()
            self.p_cols = np.repeat(np.arange(nx), np.diff(pm.indptr))
            self.solver = self.osqp.OSQP()
            self.solver.setup(P=pm, q=q, A=am, l=lower, u=upper, verbose=False,
                              eps_abs=0.015, eps_rel=0.001, max_iter=2500,
                              polishing=False, warm_starting=True)
        else:
            self.solver.update(Px=p[self.p_rows, self.p_cols], q=q,
                               Ax=a[self.a_rows, self.a_cols], l=lower, u=upper)
        if self.backend == "osqp":
            result = self.solver.solve(raise_error=False)
        self.last_solver_status = result.info.status
        self.solves += 1
        self.solve_time_ms = 1000 * (time.perf_counter() - started)
        if result.x is None or result.info.status_val not in (1, 2) or not np.all(np.isfinite(result.x)):
            return None
        return np.clip(torque_rows @ result.x + bias[self.vadr],
                       m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])

    def step(self):
        m, d = self.model, self.data
        tilt_cos = float(d.xmat[self.pelvis].reshape(3, 3)[2, 2])
        if d.qpos[2] < 0.45 or tilt_cos < math.cos(math.radians(50)):
            self.fallen = True
        if self.fallen:
            # A fall is visible; do not reset or force the base back upright.
            self.last_torque = np.clip(-3 * d.qvel[self.vadr], m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        elif d.time - self.last_update >= self.period - 1e-9:
            torque = self._solve()
            self.last_update = d.time
            if torque is None:
                self.failures += 1
                self.last_torque = np.clip(self.last_torque - 0.5 * d.qvel[self.vadr],
                                          m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
            else:
                self.last_torque = torque
        d.ctrl[:] = self.last_torque
        self.max_torque_ratio = max(self.max_torque_ratio, float(np.max(
            np.abs(d.ctrl) / np.maximum(1e-6, m.actuator_ctrlrange[:, 1]))))
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)

    def status(self):
        from physics_benchmark import measure
        measured = measure(self.model, self.data)
        state = "DOUBLE SUPPORT" if min(measured["foot_loads_n"]) > 10 else "SETTLING / SUPPORT LIMITED"
        return {"state": "FALLEN - press R to reset" if self.fallen else state,
                "fallen": self.fallen, "solver_status": self.last_solver_status,
                "qp_solves": self.solves, "qp_failures": self.failures, "solver_backend": self.backend,
                "solve_ms": self.solve_time_ms, "max_torque_limit_ratio": self.max_torque_ratio,
                "limited_joints": self.limited_joints, "com_target_m": self.com_goal.tolist(),
                "feet_planted": True, "kinematic_preview": False, **measured}
