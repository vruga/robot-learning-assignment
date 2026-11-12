"""Privileged scripted expert for move_plate (TRAINING-TIME DATA GENERATION ONLY).

Reads the simulator's true plate/rack poses, so it must never be used as an evaluated
policy. Its role: generate many varied, successful demonstrations (and DAgger labels)
that a vision+proprioception policy is then distilled from. Disclose this in the report.
Not a replay: every command is computed in closed loop from the current state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files

import mujoco
import numpy as np

LAYOUT = json.loads(files("advaitic_eval").joinpath("action_layout.json").read_text())
LOW = np.asarray(LAYOUT["action_low"])
HIGH = np.asarray(LAYOUT["action_high"])
L_ARM = slice(4, 11)  # left-arm slots in the 20-d action
R_ARM = slice(11, 18)
SHOULDER_Y = 0.147  # left end-effector y offset from pelvis when arm is at rest
PLATE_R = 0.114
PELVIS_HEIGHT_NORM = 0.13  # demos' constant pelvis-height command (~0.74 m)


def drive(err, tol, gain=1.5, vmin=0.1, vmax=0.25):
    """Velocity command with a floor: the walking controller ignores tiny commands."""
    if abs(err) < tol:
        return 0.0
    return float(np.sign(err) * np.clip(abs(gain * err), vmin, vmax))


def toward(cur, target, step):
    """Move at most `step` metres per control step toward the target."""
    delta = np.asarray(target, dtype=float) - cur
    n = np.linalg.norm(delta)
    return cur + (delta if n <= step else delta * step / n)


def norm(raw):
    return np.clip(2 * (raw - LOW) / (HIGH - LOW) - 1, -1, 1).astype(np.float32)


@dataclass
class Params:
    front_gap: float = 0.05  # hand x clearance in front of the near rim before advancing
    rim_depth: float = 0.0  # how far the pads reach past the near rim (pad half-length 33 mm)
    grasp_dz: float = 0.0  # site z above plate centre when grasping
    lift: float = 0.12
    yaw_gain: float = 0.01
    carry_vmax: float = 0.18
    plate_ahead: float = 0.085  # plate centre ahead of the hand site when pinched at the rim
    dest_dy: float = 0.01  # plate y offset to the slot site at rest (reset places +10 mm)
    dest_dz: float = -0.033  # plate z offset to the slot site at rest
    pelvis_standoff: float = 0.51  # plate_x - pelvis_x at grasp (demos: hand ~0.43 ahead of pelvis)
    ik_gain: float = 0.2
    adopt_executed: bool = True  # False (DART): keep the expert's own integrator when noise drives
    max_joint_step: float = 0.02  # rad per control step
    settle_steps: int = 25
    gripper_close_steps: int = 40
    log: bool = False
    log_every: int = 20
    log_from: int = 0
    log_to: int = 10**9
    extra: dict = field(default_factory=dict)


class Expert:
    """Phase-based, closed-loop pick-and-place using the left arm and base strafing."""

    def __init__(self, env, params: Params | None = None):
        self.env = env
        self.p = params or Params()
        self.inner = env.inner_env
        self.m = self.inner.model
        self.scratch = mujoco.MjData(self.m)
        name = "g1_29dof_with_dex1_1/left_end_effector"
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, name)
        jn = [n for n in LAYOUT["limb_names"] if n.startswith("left")]
        ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"g1_29dof_with_dex1_1/{n}")
            for n in jn
        ]
        body = self.inner.plates[0].body.id if hasattr(self.inner.plates[0], "body") else None
        pb = {
            i
            for i in range(self.m.nbody)
            if (mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i) or "").startswith("plate/")
        }
        del body
        self.plate_geoms = np.array(
            [
                g
                for g in range(self.m.ngeom)
                if self.m.geom_bodyid[g] in pb and self.m.geom_contype[g]
            ]
        )
        self.qadr = np.array([self.m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([self.m.jnt_dofadr[i] for i in ids])
        self.jlo = self.m.jnt_range[ids, 0]
        self.jhi = self.m.jnt_range[ids, 1]

    # ----- privileged readings -------------------------------------------------------------
    def plate(self):
        return np.array(self.inner.plates[0].get_position())

    def rim(self):
        """Centre of the plate's collider panel nearest the robot (the rim we pinch)."""
        d = self.inner.data
        pos = d.geom_xpos[self.plate_geoms]
        return pos[np.argmin(pos[:, 0])].copy()

    def plate_up(self):
        w, x, y, z = self.inner.plates[0].get_quaternion()
        return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])

    def plate_yaw(self):
        """Heading of the plate normal in the xy plane relative to -y (rad); 0 = ideal."""
        up = self.plate_up()
        return float(np.arctan2(up[0], -up[1]))

    def plate_angle_deg(self):
        """Angle between the plate normal and the robot's right (-y); success needs <= 20."""
        w, x, y, z = self.inner.plates[0].get_quaternion()
        up = np.array(
            [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]
        )  # rotate (0,0,1)
        return float(np.degrees(np.arccos(np.clip(-up[1], -1, 1))))

    def slots(self):
        return np.array([self.inner.data.bind(s).xpos for s in self.inner.rack_target.sites])

    def hand(self):
        return self.inner.data.site_xpos[self.site].copy()

    def hand_rot(self):
        return self.inner.data.site_xmat[self.site].reshape(3, 3).copy()

    # ----- IK ------------------------------------------------------------------------------
    def ik_step(self, goal_pos, goal_rot, q_cmd):
        """One damped-least-squares step of the left arm toward a hand pose, in world frame."""
        s = self.scratch
        s.qpos[:] = self.inner.data.qpos  # Jacobian at the ACTUAL configuration
        mujoco.mj_kinematics(self.m, s)
        mujoco.mj_comPos(self.m, s)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, s, jp, jr, self.site)
        pos = s.site_xpos[self.site]  # actual hand pose: error feedback acts as an integrator
        rot = s.site_xmat[self.site].reshape(3, 3)
        e_pos = goal_pos - pos
        # orientation error as rotation vector (world frame)
        r_err = goal_rot @ rot.T
        e_rot = 0.5 * np.array(
            [r_err[2, 1] - r_err[1, 2], r_err[0, 2] - r_err[2, 0], r_err[1, 0] - r_err[0, 1]]
        )
        J = np.vstack([jp[:, self.dadr], 0.6 * jr[:, self.dadr]])
        e = np.concatenate([e_pos, 0.6 * e_rot])
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e) * self.p.ik_gain
        step = np.clip(dq, -self.p.max_joint_step, self.p.max_joint_step)
        return (
            np.clip(q_cmd + step, self.jlo, self.jhi),
            np.linalg.norm(e_pos),
            np.linalg.norm(e_rot),
        )

    # ----- action assembly -----------------------------------------------------------------
    def action(self, state, q_cmd, vx=0.0, vy=0.0, grip=-1.0):
        raw = np.zeros(20)
        raw[0], raw[1] = vx, vy
        raw[2] = state[48]
        raw[L_ARM] = q_cmd
        raw[R_ARM] = self.q_right
        a = norm(raw)
        a[18] = grip
        a[19] = -1.0
        return a

    def step(self, a):
        ts = self.env.step(a)
        self.t += 1
        return ts

    def run(self, seed, max_steps=1700, executor=None):
        """executor(ts, expert_action) -> action to execute (DAgger); labels stay the expert's."""
        env, p = self.env, self.p
        ts = env.reset(seed=seed)
        self.t = 0
        state = ts.low_dim_obs
        self.q_right = state[9:16].copy()
        q_cmd = state[:7].copy()
        base_rot = self.hand_rot()  # reset hand orientation: fingers +x, jaws along y
        yaw_corr = 0.0
        rel0 = np.zeros(2)
        ok_count = 0
        trace = {"obs_rgb": [], "obs_state": [], "act": []}
        phase_log = []
        slots = self.slots()
        # destination: slot nearest the plate in y keeps the strafe short
        dest = slots[np.argmin(np.abs(slots[:, 1] - self.plate()[1]))]
        plate0 = self.plate().copy()
        phase = "ready"
        grip = -1.0
        phase_t = 0

        def record(ts_, a_):
            trace["obs_rgb"].append(np.asarray(ts_.rgb_obs, dtype=np.uint8))
            trace["obs_state"].append(np.asarray(ts_.low_dim_obs, dtype=np.float32))
            trace["act"].append(a_)

        while self.t < max_steps:
            state = ts.low_dim_obs
            pelvis = state[18:20]
            vel = np.linalg.norm(state[22:24])
            plate = self.plate()
            hand = self.hand()
            vx = vy = 0.0
            ready = np.array([pelvis[0] + 0.28, max(plate[1], pelvis[1] + 0.18), plate[2] + 0.22])
            if phase == "ready":
                goal = ready
                if (np.linalg.norm(hand - goal) < 0.03 and phase_t > 5) or phase_t > 90:
                    phase, phase_t = "walk", 0
            if phase == "walk":
                ex = (plate[0] - p.pelvis_standoff) - pelvis[0]
                ey = (plate[1] - SHOULDER_Y) - pelvis[1]
                vx = drive(ex, 0.04)
                vy = drive(ey, 0.02)
                goal = ready
                if (abs(ex) < 0.04 and abs(ey) < 0.02 and phase_t > 10) or phase_t > 400:
                    phase, phase_t = "settle", 0
            if phase == "settle":
                goal = ready
                if phase_t >= p.settle_steps:
                    phase, phase_t = "pregrasp", 0
            rim = self.rim()
            if phase == "pregrasp":  # in front of the near rim, a bit above, jaws straddling y
                goal = np.array([rim[0] - p.front_gap, rim[1], rim[2] + 0.10])
                if (np.linalg.norm(hand - goal) < 0.015 and phase_t > 5) or phase_t > 150:
                    phase, phase_t = "descend", 0
            if phase == "descend":
                goal = np.array([rim[0] - p.front_gap, rim[1], rim[2] + p.grasp_dz])
                if (np.linalg.norm(hand - goal) < 0.010 and phase_t > 5) or phase_t > 150:
                    phase, phase_t = "advance", 0
            if phase == "advance":  # slide the open jaws over the near rim
                goal = np.array([rim[0] + p.rim_depth, rim[1], rim[2] + p.grasp_dz])
                if (np.linalg.norm(hand - goal) < 0.012 and phase_t > 5) or phase_t > 60:
                    phase, phase_t = "close", 0
            if phase == "close":
                goal = None  # hold: freeze the joint command
                grip = 1.0
                if phase_t >= p.gripper_close_steps:
                    rel0 = hand[:2] - pelvis[:2]  # hold this hand-to-pelvis offset while carrying
                    phase, phase_t = "lift", 0
            if phase in ("lift", "carry", "steady", "align", "place"):
                # plate pivots in a 66 mm pinch and the torso sways: close the loop on plate yaw
                yaw_corr = float(np.clip(yaw_corr - p.yaw_gain * self.plate_yaw(), -0.6, 0.6))
            if phase == "lift":
                grip = 1.0
                goal = np.array([*(pelvis + rel0), plate0[2] + p.grasp_dz + p.lift])
                if hand[2] > plate0[2] + p.grasp_dz + p.lift - 0.015 or phase_t > 120:
                    phase, phase_t = "carry", 0
            carry_z = plate0[2] + p.grasp_dz + p.lift
            gx, gy = dest[0], dest[1] + p.dest_dy
            if phase == "carry":  # coarse: strafe the base, hand anchored to the pelvis
                ey = gy - plate[1]
                vy = drive(ey, 0.05, gain=1.5, vmin=0.1, vmax=p.carry_vmax)
                # also centre the arm's reach on the slot in x (racks shift +-5 cm)
                pelvis_goal_x = gx - p.plate_ahead - rel0[0]
                vx = drive(pelvis_goal_x - pelvis[0], 0.03, vmax=p.carry_vmax)
                goal = np.array([pelvis[0] + rel0[0], pelvis[1] + rel0[1], carry_z])
                if abs(ey) < 0.05 and abs(pelvis_goal_x - pelvis[0]) < 0.05 and phase_t > 15:
                    phase, phase_t = "steady", 0
            if phase == "steady":  # base stopped: let the gait sway damp
                goal = np.array([pelvis[0] + rel0[0], pelvis[1] + rel0[1], carry_z])
                if phase_t >= 40 and vel < 0.03:
                    phase, phase_t = "align", 0
                    ok_count = 0
            if phase == "align":  # fine: arm moves the plate onto the slot (feedback on plate pose)
                err = np.array([gx - plate[0], gy - plate[1]])
                step_xy = np.clip(err, -0.008, 0.008)
                goal = np.array([hand[0] + step_xy[0], hand[1] + step_xy[1], carry_z])
                ok_count = ok_count + 1 if np.abs(err).max() < 0.008 else 0
                if ok_count >= 10:
                    phase, phase_t = "place", 0
                elif phase_t > 200:
                    phase, phase_t = "place", 0
            if phase == "place":
                grip = 1.0
                ez = (dest[2] + p.dest_dz) - plate[2]
                goal = hand + np.array(
                    [
                        np.clip(dest[0] - plate[0], -0.005, 0.005),
                        np.clip(dest[1] + p.dest_dy - plate[1], -0.005, 0.005),
                        np.clip(ez, -0.006, 0.006),
                    ]
                )
                if abs(ez) < 0.006 or phase_t > 150:
                    phase, phase_t = "release", 0
            if phase == "release":
                goal = None  # hold: freeze the joint command
                grip = -1.0
                if phase_t >= 30:
                    phase, phase_t = "retreat", 0
            if phase == "retreat":
                grip = -1.0
                goal = toward(hand, [hand[0] - 0.01, hand[1], hand[2]], 0.004)
                if phase_t >= 40:
                    phase, phase_t = "wait", 0
            if phase == "wait":
                goal = None  # hold: freeze the joint command
                grip = -1.0
            if goal is None:
                ep = er = 0.0
            else:
                c, sn = np.cos(yaw_corr), np.sin(yaw_corr)
                rz = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
                q_cmd, ep, er = self.ik_step(goal, rz @ base_rot, q_cmd)
            a = self.action(state, q_cmd, vx, vy, grip)
            record(ts, a)
            a_exec = a if executor is None else executor(ts, a)
            if (
                a_exec is not a and p.adopt_executed
            ):  # someone else drove: the expert's command follows the executed one
                q_cmd = (LOW + (np.asarray(a_exec, dtype=float) + 1) / 2 * (HIGH - LOW))[L_ARM]
            ts = self.step(a_exec)
            phase_t += 1
            if p.log and self.t % p.log_every == 0 and p.log_from <= self.t < p.log_to:
                print(
                    f"t={self.t:4d} {phase:9s} pelvis={np.round(pelvis, 3)} hand={np.round(hand, 3)} "
                    f"plate={np.round(plate, 3)} ep={ep:.3f} er={er:.3f} ang={self.plate_angle_deg():.1f} yaw={np.degrees(self.plate_yaw()):.1f} grip={grip} gst={np.round(state[44:46], 2)} fing={np.round(state[7:9], 3)} q={np.round(q_cmd, 2)}",
                    flush=True,
                )
            phase_log.append(phase)
            if self.env.episode_succeeded() and phase in ("release", "retreat", "wait"):
                if phase_t > 60:
                    break
            if self.env.episode_fell() or (executor is not None and self.inner._fail()):
                break
        success = bool(self.env.episode_succeeded())
        return {
            "success": success,
            "fell": bool(self.env.episode_fell()),
            "steps": self.t,
            "final_phase": phase,
            "trace": {k: np.asarray(v) for k, v in trace.items()},
        }
