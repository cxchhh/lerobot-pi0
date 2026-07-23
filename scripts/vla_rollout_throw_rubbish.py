"""Minimal VLA rollout for the bfm_g1_throw_rubbish dataset.

Cuts play_vla_infer.py down to: G1 sim (no room/cube/sofa) + BFM tracker +
Websocket VLA server + one external "track" camera dumped to mp4.  Uses a
LeRobot test-set episode's anchor_pose[0] to place the robot at the right
world xy/yaw; joints start from build_init_qpos_template (default stance)
because 29-dof joint angles aren't reconstructable from the 38-dim state.

Run:
  # Terminal 1
  bash run_server_throw_rubbish.sh
  # Terminal 2
  python scripts/vla_rollout_throw_rubbish.py --episode-idx 0 \
      --out-mp4 /tmp/rollout_ep0.mp4 --max-steps 700
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import einops  # noqa: F401  (imported for parity with server-side image pipeline)
import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from locomanip import tracking_constants as consts
from locomanip.bfm.infer import BFMRawInferFn as BFMInferFn
from locomanip.bfm.scaletrack import BODY_NAMES, DEFAULT_JOINT_POS, MODE_CANDIDATES
from locomanip.deploy.infer_common import (
    MODE_NAME, GRIP_IDX, ChunkBuffer, make_future_idx,
)
from locomanip.utils.dex3 import finger_ctrl, setup_dex3_fingers
from locomanip.utils.init_qpos import build_init_qpos_template
from locomanip.utils.pose_math import (
    EE_LINK_NAMES, T_world2nav, body_quat_wxyz_from_xmat, quat_to_rot6d,
)
from locomanip.utils.nav_math import (
    nav_to_world_pos, nav_to_world_quat, rot6d_to_quat_wxyz,
)
from gx_infer.websocket_client_policy import WebsocketClientPolicy

# LeRobot dataset (already in lerobot conda env)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


@dataclass
class Args:
    ckpt_dir: str = ("outputs/train/2026-07-14/00-19-27_bfm_g1_throw_rubbish"
                     "/checkpoints/100000/pretrained_model")
    """(Informational only — the policy comes from the running server.)"""
    test_root: str = ("/mnt/kpfs/chenxuchuan/sandbox/G1-VLA/"
                      "bfm_g1_throw_rubbish_lerobot21_threecam_nvenc_train_test/"
                      "test/bfm_g1_throw_rubbish")
    bfm_ckpt: str = "/mnt/kpfs/chenxuchuan/sandbox/locomanip/ckpts/bfm_g1_29dof_raw.pt"
    host: str = "localhost"
    port: int = 8002
    episode_idx: int = 0
    out_mp4: str = "/tmp/rollout_throw_rubbish.mp4"
    max_steps: int = 700
    cam_h: int = 720
    cam_w: int = 1280
    render_fps: int = 25
    action_stride: int = 3
    rtc_prefix_len: int = 3
    freq: int = 50
    sim_dt: float = 0.002
    warmup_seconds: float = 1.0
    n_obs_states: int = 2
    task: str = "throw rubbish into the trash bin"
    device: str = "cuda:0"
    teacher_force: bool = False
    """If True, at every tick feed the policy the DATASET's own state +
    real-camera images at frame `ts`, instead of sim-derived obs.  The
    predicted action still drives BFM -> sim.  Diagnostic: if this looks
    right, the policy is fine and sim/vision covariate shift is the issue."""


# ---- state-build FSM constants (copied from play_vla_infer.py) ----
Z_IN_LO_C, V_IN_LO_C = 0.100, 0.002
Z_OUT_HI_C, V_OUT_HI_C = 0.115, 0.006
HOLD_FRAMES_C = 12


def main():
    args = tyro.cli(Args)
    ctrl_dt = 1.0 / args.freq
    num_substeps = int(round(ctrl_dt / args.sim_dt))
    n_warmup = int(round(args.warmup_seconds / ctrl_dt))
    render_every = max(1, args.freq // args.render_fps)

    # -------- LeRobot test-set episode --------
    ds = LeRobotDataset("bfm_g1_throw_rubbish", root=args.test_root)
    if args.episode_idx >= ds.num_episodes:
        raise SystemExit(f"episode_idx {args.episode_idx} >= {ds.num_episodes}")
    fi = int(ds.episode_data_index["from"][args.episode_idx].item())
    ti = int(ds.episode_data_index["to"][args.episode_idx].item())
    ep_len = ti - fi
    sample0 = ds[fi]
    anchor0 = sample0["observation.anchor_pose"].numpy().astype(np.float64)  # (7,) pos+quat_wxyz
    ep_task = sample0.get("task", args.task)
    print(f"[rollout] episode {args.episode_idx}: len={ep_len} frames  "
          f"anchor0 pos={anchor0[:3]} quat={anchor0[3:]}  task={ep_task!r}")

    # -------- Sim init (no room, no cube) --------
    g1_xml = str(consts.TRACK_DEX3_XML)
    mj_model = mujoco.MjModel.from_xml_path(g1_xml)
    mj_model.opt.timestep = args.sim_dt
    mj_data = mujoco.MjData(mj_model)
    fk_data = mujoco.MjData(mj_model)

    canonical = list(DEFAULT_JOINT_POS.keys())
    qadr29 = np.array([mj_model.jnt_qposadr[mj_model.joint(n).id] for n in canonical], dtype=np.int64)
    vadr29 = np.array([mj_model.jnt_dofadr[mj_model.joint(n).id] for n in canonical], dtype=np.int64)
    act29 = np.array([mj_model.actuator(n).id for n in canonical], dtype=np.int64)
    jidx29 = vadr29 - 6
    ee_body_ids = np.array([mj_model.body(n).id for n in EE_LINK_NAMES], dtype=np.int64)
    pelvis_body_id = mj_model.body("pelvis").id
    ee_idx_g1 = [BODY_NAMES.index(n) for n in EE_LINK_NAMES]
    viz_idx = [BODY_NAMES.index(n) for n in MODE_CANDIDATES[MODE_NAME]]
    nb = len(BODY_NAMES)

    # Sim cameras (in-XML head/left_wrist/right_wrist).
    cam_id_head = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam")
    cam_id_lwrist = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "left_wrist_cam")
    cam_id_rwrist = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "right_wrist_cam")
    if min(cam_id_head, cam_id_lwrist, cam_id_rwrist) < 0:
        raise SystemExit(f"cam missing (ids head={cam_id_head} lw={cam_id_lwrist} rw={cam_id_rwrist})")
    # enlarge offscreen framebuffer so the external cam can be up to 1920x1080
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, args.cam_w)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, args.cam_h)
    # policy-input cams: match the dataset feature size (3x140x224)
    POL_H, POL_W = 140, 224
    rend_head = mujoco.Renderer(mj_model, height=POL_H, width=POL_W)
    rend_lwrist = mujoco.Renderer(mj_model, height=POL_H, width=POL_W)
    rend_rwrist = mujoco.Renderer(mj_model, height=POL_H, width=POL_W)
    # external "track" camera (mp4 output)
    cam_id_track = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    if cam_id_track < 0:
        raise SystemExit("track camera missing in XML")
    rend_ext = mujoco.Renderer(mj_model, height=args.cam_h, width=args.cam_w)

    # -------- Initial qpos: default stance + snap pelvis xy/yaw to anchor0 --------
    # We START from the default stance but immediately let BFM warmup track a
    # hold-motion whose 4 EE targets = dataset t=0 EE poses in world frame.
    # After ~1s of warmup the robot converges to the dataset's initial pose
    # (arms/legs where they should be), then VLA takes over.  Root xy/yaw
    # comes from anchor_pose[0] (pelvis world pos+quat).
    init_qpos = build_init_qpos_template(mj_model).copy()
    mj_data.qpos[:] = init_qpos
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = 0.0
    mj_data.qpos[0:3] = anchor0[:3]
    mj_data.qpos[3:7] = anchor0[3:7]
    mujoco.mj_forward(mj_model, mj_data)
    fk_data.qpos[:] = mj_data.qpos.copy()
    mujoco.mj_forward(mj_model, fk_data)
    pp = mj_data.xpos[pelvis_body_id]
    print(f"[rollout] pelvis spawn: ({pp[0]:+.3f},{pp[1]:+.3f},{pp[2]:.3f})")

    # ---- decode dataset t=0 EE local pose -> world for BFM warmup ref ----
    # state[0:36] = 4 EE (l_wrist, r_wrist, l_ankle, r_ankle) local xyz+rot6d,
    # expressed in the pelvis-yaw-only + on-ground local frame (see
    # play_vla_infer.py:1088-1122).  Convert back to world using anchor0's
    # pelvis pose.
    state0 = sample0["observation.state"].numpy().astype(np.float64)  # (38,)
    ee_local = state0[:36].reshape(4, 9)
    ee_pos_local = ee_local[:, :3]                                    # (4, 3)
    ee_rot6d_local = ee_local[:, 3:9]                                 # (4, 6)
    ee_quat_local = rot6d_to_quat_wxyz(ee_rot6d_local)                # (4, 4) wxyz
    pel_pos0_w = anchor0[:3].copy()
    pel_quat0_w = anchor0[3:7].copy()  # wxyz
    # yaw from pelvis forward projected on ground plane (matches build path
    # in the per-tick state).
    from scipy.spatial.transform import Rotation as _R
    pel_R0 = _R.from_quat(pel_quat0_w[[1, 2, 3, 0]]).as_matrix()      # xyzw for scipy
    pf0 = pel_R0[:, 0]
    pf0_gp = np.array([pf0[0], pf0[1], 0.0])
    pel_yaw0 = float(np.arctan2(pf0_gp[1], pf0_gp[0])) if np.linalg.norm(pf0_gp) > 1e-9 else 0.0
    pel_xy_ground0 = np.array([pel_pos0_w[0], pel_pos0_w[1], 0.0])
    hold_ee_pos = nav_to_world_pos(ee_pos_local, pel_xy_ground0, pel_yaw0).astype(np.float32)
    hold_ee_quat = nav_to_world_quat(ee_quat_local, pel_yaw0).astype(np.float32)
    print(f"[rollout] hold EE (world) l_wrist={hold_ee_pos[0]} r_wrist={hold_ee_pos[1]}")
    print(f"[rollout]                  l_ankle={hold_ee_pos[2]} r_ankle={hold_ee_pos[3]}")

    # -------- BFM + hold-motion + dex3 --------
    eff_stride = max(1, args.action_stride)
    future_idx = make_future_idx(eff_stride)
    print(f"[rollout] action_stride={eff_stride}  FUTURE_IDX={future_idx}")
    infer_fn = BFMInferFn(model_path=args.bfm_ckpt, mj_model=mj_model,
                          mode_name=MODE_NAME, ctrl_dt=ctrl_dt,
                          future_idx=future_idx, device=args.device)
    kp29 = infer_fn.kp[jidx29].astype(np.float32)
    kd29 = infer_fn.kd[jidx29].astype(np.float32)
    fingers = setup_dex3_fingers(mj_model, canonical)
    infer_fn.reset()
    infer_fn.warm_start(mj_data)

    # warmup hold motion: BFM tracks dataset t=0 EE targets (computed above),
    # NOT the FK'd default-stance EE, so warmup drives the robot AWAY from
    # A-pose and INTO the dataset's initial arm/leg configuration.
    n_total = args.max_steps if args.max_steps > 0 else ep_len
    T_hold = n_total + n_warmup + 32

    class _Motion: pass
    motion = _Motion()
    motion.body_pos_w = np.zeros((T_hold, nb, 3), dtype=np.float32)
    motion.body_quat_w = np.zeros((T_hold, nb, 4), dtype=np.float32)
    motion.body_quat_w[..., 0] = 1.0
    for _i, bi in enumerate(viz_idx):
        motion.body_pos_w[:, bi, :] = hold_ee_pos[_i]
        motion.body_quat_w[:, bi, :] = hold_ee_quat[_i]
    motion.joint_pos = np.zeros((T_hold, 29), dtype=np.float32)
    motion.joint_vel = np.zeros((T_hold, 29), dtype=np.float32)
    motion.body_lin_vel_w = None
    motion.body_ang_vel_w = None
    motion.num_frames = T_hold
    motion.gripper = None

    # VLA ring
    ring_T = T_hold
    ring = _Motion()
    ring.body_pos_w = np.zeros((ring_T, nb, 3), dtype=np.float32)
    ring.body_quat_w = np.zeros((ring_T, nb, 4), dtype=np.float32)
    ring.body_quat_w[..., 0] = 1.0
    ring.joint_pos = np.zeros((ring_T, 29), dtype=np.float32)
    ring.joint_vel = np.zeros((ring_T, 29), dtype=np.float32)
    ring.body_lin_vel_w = None
    ring.body_ang_vel_w = None
    ring.num_frames = ring_T
    ring.gripper = None

    # -------- VLA server --------
    print(f"[rollout] connecting to ws://{args.host}:{args.port}")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"[rollout] server metadata: {policy.get_server_metadata()}")
    chunk_buf = ChunkBuffer(policy, future_offsets=future_idx,
                            action_stride=eff_stride, freq=args.freq,
                            rtc_prefix_len=args.rtc_prefix_len,
                            block=False, prefetch_mode="deferred",
                            chunk_exec_ticks=0)

    finger_grip = np.zeros(2, dtype=np.float32)
    finger_grip[0] = 1.0   # left hand always closed (dataset convention)
    first_policy_tick = True

    state_hist: deque = deque(maxlen=max(1, args.n_obs_states * eff_stride))
    anchor_hist: deque = deque(maxlen=max(1, args.n_obs_states * eff_stride))
    l_contact_s, r_contact_s = True, True
    l_hold_s, r_hold_s = 0, 0
    prev_l_ankle_w = None
    prev_r_ankle_w = None

    frames_ext = []

    # -------- Control loop (mirrors play_vla_infer.py:1060-1602) --------
    for ts in range(n_total):
        is_warmup = ts < n_warmup

        # per-tick state (copied from play_vla_infer.py:1071-1170)
        mujoco.mj_forward(mj_model, mj_data)
        fk_data.qpos[:] = mj_data.qpos.copy()
        mujoco.mj_forward(mj_model, fk_data)
        pelvis_pos_w = fk_data.xpos[pelvis_body_id].astype(np.float64).copy()
        pelvis_R_w = fk_data.xmat[pelvis_body_id].reshape(3, 3).astype(np.float64).copy()
        pelvis_quat_w = body_quat_wxyz_from_xmat(fk_data.xmat[pelvis_body_id]).astype(np.float64)
        ee_pos_w = fk_data.xpos[ee_body_ids].astype(np.float64)
        ee_R_w = np.stack(
            [fk_data.xmat[i].reshape(3, 3) for i in ee_body_ids], axis=0
        ).astype(np.float64)

        pf_w = pelvis_R_w[:, 0]
        pf_gp = np.array([pf_w[0], pf_w[1], 0.0])
        _n = float(np.linalg.norm(pf_gp))
        pelvis_yaw_gv = float(np.arctan2(pf_gp[1], pf_gp[0])) if _n > 1e-9 else 0.0
        _cy, _sy = np.cos(pelvis_yaw_gv), np.sin(pelvis_yaw_gv)
        R_w2l = np.array([[_cy, _sy, 0.0], [-_sy, _cy, 0.0], [0.0, 0.0, 1.0]])
        pelvis_xy_ground = pelvis_pos_w.copy()
        pelvis_xy_ground[2] = 0.0
        ee_pos_local = (ee_pos_w - pelvis_xy_ground) @ R_w2l.T
        ee_R_local = np.einsum('ij,bjk->bik', R_w2l, ee_R_w)
        _q_xyzw = R.from_matrix(ee_R_local).as_quat()
        ee_quat_local = _q_xyzw[:, [3, 0, 1, 2]]
        ee_rot6d = quat_to_rot6d(ee_quat_local)
        state_ee = np.concatenate([ee_pos_local, ee_rot6d], axis=-1).reshape(-1).astype(np.float32)

        # contact FSM (dataset has contact_in_state = True)
        lz_s = float(ee_pos_w[2, 2]); rz_s = float(ee_pos_w[3, 2])
        if prev_l_ankle_w is None:
            lv_s, rv_s = 0.0, 0.0
        else:
            lv_s = float(np.linalg.norm(ee_pos_w[2] - prev_l_ankle_w))
            rv_s = float(np.linalg.norm(ee_pos_w[3] - prev_r_ankle_w))
        if l_hold_s > 0: l_hold_s -= 1
        if r_hold_s > 0: r_hold_s -= 1
        if l_hold_s == 0:
            want_l_air = (lz_s > Z_OUT_HI_C or lv_s > V_OUT_HI_C) if l_contact_s \
                         else not (lz_s < Z_IN_LO_C and lv_s < V_IN_LO_C)
            if l_contact_s and want_l_air:
                l_contact_s = False; l_hold_s = HOLD_FRAMES_C
            elif (not l_contact_s) and (not want_l_air):
                l_contact_s = True; l_hold_s = HOLD_FRAMES_C
        if r_hold_s == 0:
            want_r_air = (rz_s > Z_OUT_HI_C or rv_s > V_OUT_HI_C) if r_contact_s \
                         else not (rz_s < Z_IN_LO_C and rv_s < V_IN_LO_C)
            if r_contact_s and want_r_air:
                r_contact_s = False; r_hold_s = HOLD_FRAMES_C
            elif (not r_contact_s) and (not want_r_air):
                r_contact_s = True; r_hold_s = HOLD_FRAMES_C
        if (not l_contact_s) and (not r_contact_s):
            if (lz_s + lv_s) <= (rz_s + rv_s):
                l_contact_s = True
            else:
                r_contact_s = True
        prev_l_ankle_w = ee_pos_w[2].copy()
        prev_r_ankle_w = ee_pos_w[3].copy()
        state = np.concatenate([state_ee,
                                np.array([float(l_contact_s), float(r_contact_s)],
                                         dtype=np.float32)])
        state_hist.append(state)
        anchor_hist.append(np.concatenate([pelvis_pos_w, pelvis_quat_w]).astype(np.float32))

        if is_warmup:
            active_motion = motion
            active_ts = ts
            finger_grip[1] = 0.0
        else:
            if first_policy_tick:
                last = state_hist[-1]; last_anc = anchor_hist[-1]
                state_hist.clear(); anchor_hist.clear()
                state_hist.append(last); anchor_hist.append(last_anc)

            def _make_obs(_first=[first_policy_tick]):
                if _first[0] and hasattr(policy, "reset"):
                    policy.reset()
                first_was = _first[0]; _first[0] = False
                if args.teacher_force:
                    # feed dataset frame ts directly (no sim leak).  ts
                    # counts sim ticks INCLUDING warmup; dataset frame
                    # index = ts - n_warmup so t=0 aligns.
                    dset_t = min(max(0, ts - n_warmup), ep_len - 1)
                    ss = ds[fi + dset_t]
                    n_states = max(1, args.n_obs_states)
                    st_1 = ss["observation.state"].numpy().astype(np.float32)  # (38,)
                    state_stack = np.broadcast_to(st_1[None], (n_states, 38)).copy()
                    anc = ss["observation.anchor_pose"].numpy().astype(np.float32)  # (7,)
                    def _to_uint8(t):
                        return (t.numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                    return {
                        "observation.state": state_stack,
                        "observation.anchor_pose": anc,
                        "observation.images.head": _to_uint8(ss["observation.images.head"]),
                        "observation.images.left_wrist": _to_uint8(ss["observation.images.left_wrist"]),
                        "observation.images.right_wrist": _to_uint8(ss["observation.images.right_wrist"]),
                        "task": ss.get("task", ep_task),
                        "reset": 1 if first_was else 0,
                    }
                hist = list(state_hist); anc_hist = list(anchor_hist)
                n_states = max(1, args.n_obs_states)
                stride = max(1, eff_stride)
                needed = (n_states - 1) * stride + 1
                while len(hist) < needed:
                    hist.insert(0, hist[0]); anc_hist.insert(0, anc_hist[0])
                pick_idx = [-1 - (n_states - 1 - i) * stride for i in range(n_states)]
                state_stack = np.stack([hist[k] for k in pick_idx], axis=0).astype(np.float32)
                rend_head.update_scene(mj_data, cam_id_head)
                rend_lwrist.update_scene(mj_data, cam_id_lwrist)
                rend_rwrist.update_scene(mj_data, cam_id_rwrist)
                rgb_head = rend_head.render().copy()          # (POL_H, POL_W, 3) uint8
                rgb_lwrist = rend_lwrist.render().copy()
                rgb_wrist = rend_rwrist.render().copy()
                anc = np.concatenate([pelvis_pos_w, pelvis_quat_w]).astype(np.float32)
                return {
                    "observation.state": state_stack,
                    "observation.anchor_pose": anc,
                    "observation.images.head": rgb_head,
                    "observation.images.left_wrist": rgb_lwrist,
                    "observation.images.right_wrist": rgb_wrist,
                    "task": ep_task,
                    "reset": 1 if first_was else 0,
                }

            future_actions, chunk_anchor = chunk_buf.query(_make_obs)
            if first_policy_tick:
                print(f"[rollout] >>> VLA takeover @ t={ts}")
                first_policy_tick = False

            chunk_anchor_pos = np.asarray(chunk_anchor[:3], dtype=np.float64)
            chunk_anchor_yaw = float(np.arctan2(
                2 * (chunk_anchor[3] * chunk_anchor[6] + chunk_anchor[4] * chunk_anchor[5]),
                1 - 2 * (chunk_anchor[5] ** 2 + chunk_anchor[6] ** 2),
            ))

            # unwrap 6 future EEs, world frame
            fut_pos_w_all = np.zeros((6, 4, 3), dtype=np.float64)
            fut_quat_w_all = np.zeros((6, 4, 4), dtype=np.float64)
            sim_ee_world_now = fk_data.xpos[ee_body_ids].astype(np.float64)
            GROUND_L = sim_ee_world_now[2, 2]; GROUND_R = sim_ee_world_now[3, 2]
            SWING_MAX = 0.15
            for fi_ in range(6):
                pose = future_actions[fi_, :36].reshape(4, 9)
                pos_nav = pose[:, :3].astype(np.float64)
                quat_nav = rot6d_to_quat_wxyz(pose[:, 3:9].astype(np.float64))
                pos_w = nav_to_world_pos(pos_nav, chunk_anchor_pos, chunk_anchor_yaw)
                quat_w = nav_to_world_quat(quat_nav, chunk_anchor_yaw)
                pos_w[2, 2] = min(max(pos_w[2, 2], GROUND_L), GROUND_L + SWING_MAX)
                pos_w[3, 2] = min(max(pos_w[3, 2], GROUND_R), GROUND_R + SWING_MAX)
                fut_pos_w_all[fi_] = pos_w
                fut_quat_w_all[fi_] = quat_w

            # write ring at future_idx slots
            if ts + max(future_idx) + 1 > ring.body_pos_w.shape[0]:
                grow = max(512, ts + 64)
                for arr_name, dtyp in [("body_pos_w", np.float32),
                                        ("body_quat_w", np.float32)]:
                    old = getattr(ring, arr_name)
                    new = np.zeros((grow,) + old.shape[1:], dtype=dtyp)
                    if arr_name == "body_quat_w":
                        new[..., 0] = 1.0
                    new[:old.shape[0]] = old
                    setattr(ring, arr_name, new)
                ring.joint_pos = np.zeros((grow, 29), dtype=np.float32)
                ring.joint_vel = np.zeros((grow, 29), dtype=np.float32)
                ring.num_frames = grow
            for fi_ in range(6):
                slot = ts + future_idx[fi_]
                for k, bi in enumerate(ee_idx_g1):
                    ring.body_pos_w[slot, bi] = fut_pos_w_all[fi_, k]
                    ring.body_quat_w[slot, bi] = fut_quat_w_all[fi_, k]

            active_motion = ring
            active_ts = ts
            finger_grip[1] = float(np.clip(future_actions[0, GRIP_IDX], 0.0, 1.0))

        # BFM -> PD -> sim
        joint_targets = infer_fn.infer(mj_data, active_motion, active_ts)
        for _ in range(num_substeps):
            tq = kp29 * (joint_targets - mj_data.qpos[qadr29]) + kd29 * (-mj_data.qvel[vadr29])
            tq = np.clip(tq, -consts.TORQUE_LIMIT, consts.TORQUE_LIMIT)
            mj_data.ctrl[act29] = tq
            finger_ctrl(mj_data, fingers, finger_grip)
            mujoco.mj_step(mj_model, mj_data)

        # render external mp4 frame (subsampled)
        if ts % render_every == 0:
            rend_ext.update_scene(mj_data, cam_id_track)
            frames_ext.append(rend_ext.render().copy())

        if (ts + 1) % 50 == 0:
            print(f"[rollout] tick {ts+1}/{n_total}  frames={len(frames_ext)}")

    print(f"[rollout] writing {len(frames_ext)} frames -> {args.out_mp4}")
    Path(args.out_mp4).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out_mp4, frames_ext, fps=args.render_fps,
                     codec="libx264", quality=8)
    print("[rollout] done")


if __name__ == "__main__":
    main()
