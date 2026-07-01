"""LeRobot item_transform hooks for chunk re-anchoring.

Transforms, picked via DatasetConfig.item_transform_path:

  chunk_transforms:reanchor_chunk_to_first_frame
      bfm-v1.2 layout.  action[t] is already in nav(t); re-projects every
      frame into chunk[0]'s pelvis-nav anchor.  shape preserved (39 -> 39).

  chunk_transforms:reanchor_chunk_and_reproject_state_history
      bfm-v1.2 + state-history layout.  Same action reanchor as above
      AND reprojects every history state row from its own nav(t-k) to the
      current nav(t).  Requires factory.py to stack observation.anchor_pose
      at union(state_delta_indices, action_delta_indices) so the per-row
      anchors for state are also present.

  chunk_transforms:reanchor_chunk_keep_state_local
      bfm-v1.7 layout.  Action chunk reanchored to chunk[0]'s nav (same
      as v1.2/v1.6) but state history is LEFT IN ITS OWN per-row nav --
      each stacked state row stays in nav(t-k) instead of being projected
      to nav(t).  Lets deploy skip the global-anchor reproject step:
      just sample state in current nav each tick and stack.  Pi0 sees a
      "per-frame ego-relative" history, which is enough to detect step
      phase from EE relative motion without needing absolute pelvis
      displacement (which v1.6 conveyed implicitly via the reproject).

  chunk_transforms:project_world_action_to_self_pelvis_nav
      bfm-v1.3 layout.  action[t,:36] is ref EE in WORLD; projects each
      frame to its own pelvis-nav (nav(t)).  Pelvis velocity (last 3 dims)
      is already in nav(t) at write time so it's passed through.  Contact
      / grip untouched.  shape preserved (42 -> 42).

  chunk_transforms:reanchor_chunk_keep_state_local_state_dropout
      Same as `reanchor_chunk_keep_state_local` plus train-time state
      dropout.  Takes a `state_dropout_p` kwarg (default 0.0 = off);
      set via DatasetConfig.item_transform_kwargs in the training cfg.
      Factory binds the kwargs only on the train dataset, so the test
      dataset (and deploy, which bypasses LeRobotDataset entirely)
      naturally see the unaugmented transform.
"""
from __future__ import annotations

import torch


def _yaw_from_wxyz(q: torch.Tensor) -> torch.Tensor:
    """(.., 4) wxyz -> (..) yaw."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _R2_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """(...,) -> (..., 2, 2) world->nav rotation (R^T of yaw rotation)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    # nav x-axis = pelvis forward (yaw); apply_T_pos uses R^T (world->nav)
    # [[c, s], [-s, c]]
    return torch.stack([
        torch.stack([c, s], dim=-1),
        torch.stack([-s, c], dim=-1),
    ], dim=-2)


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product (.., 4) wxyz."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def _rot6d_to_mat(rot6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) right-handed orthonormal."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-12)
    a2_p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2_p / (a2_p.norm(dim=-1, keepdim=True) + 1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _mat_to_rot6d(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6)."""
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def _rot6d_to_mat_writer(rot6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) using the bfm_track_to_lerobot writer layout:
      rot6d = m[:3, :2].reshape(-1)  (row-major interleave)
      -> rot6d = [m[0,0], m[0,1], m[1,0], m[1,1], m[2,0], m[2,1]]
      -> col0 = rot6d[..., 0::2]  (3 entries: m[0,0], m[1,0], m[2,0])
      -> col1 = rot6d[..., 1::2]  (3 entries: m[0,1], m[1,1], m[2,1])
    Gram-Schmidt to produce an orthonormal R = [b1, b2, b3].

    The OTHER `_rot6d_to_mat` in this module takes [first-3]/[last-3] as
    col0/col1; that layout is what reanchor_chunk_to_first_frame has been
    using (and what v1.4 was trained against).  Don't touch that path --
    the v1.4 checkpoint learned the scrambled mapping; switching would
    break it.  Use *this* helper for walk-v1 and any new dataset to keep
    the writer/transform/deploy decoder all in the same layout.
    """
    a1 = rot6d[..., 0::2]
    a2 = rot6d[..., 1::2]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-12)
    a2_p = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2_p / (a2_p.norm(dim=-1, keepdim=True) + 1e-12)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def _mat_to_rot6d_writer(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6) using the writer layout (row-major
    interleave of m[:, :2]).  Inverse of _rot6d_to_mat_writer."""
    # m[..., :, :2] -> (..., 3, 2); flatten last two -> row-major interleave.
    return m[..., :, :2].flatten(start_dim=-2)


def _mat_to_quat_wxyz(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 4) wxyz.  Batched safe."""
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    out = torch.zeros(*m.shape[:-2], 4, dtype=m.dtype, device=m.device)
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-12)) * 2
    out[..., 0] = 0.25 * s
    out[..., 1] = (m[..., 2, 1] - m[..., 1, 2]) / s
    out[..., 2] = (m[..., 0, 2] - m[..., 2, 0]) / s
    out[..., 3] = (m[..., 1, 0] - m[..., 0, 1]) / s
    return out


def reanchor_chunk_to_first_frame(item: dict) -> dict:
    """Re-express the chunk's `action` so every frame's 4-EE pose is in the
    SAME pelvis-nav anchor (= chunk[0]'s anchor).  Action layout per frame
    is EE-major (each ee gets 9 contiguous cols = pos(3) + rot6d(6)):
      [0:9]    = l_wrist  (pos + rot6d)
      [9:18]   = r_wrist
      [18:27]  = l_ankle
      [27:36]  = r_ankle
      [36:38]  = l/r contact (binary)
      [38]     = right_hand_close
    Only the 4-EE pose channels are rewritten; contact/grip unchanged.
    """
    action = item["action"]                # (H, 39)
    anchor = item["observation.anchor_pose"]  # (>=H, 7): pelvis world [pos, quat wxyz]
    if action.ndim != 2 or action.shape[1] < 36:
        return item
    H = action.shape[0]
    if anchor.shape[0] < H:
        return item
    # When the policy stacks state history, factory.py expands anchor_pose
    # to the union of state_delta_indices + action_delta_indices.  The last
    # H rows correspond to the action chunk (sorted offsets, state ends at
    # 0 = first action offset).  Slice them out so the math below is
    # unchanged regardless of which layout was used.
    anchor = anchor[-H:]

    pel_xy = anchor[:, :2]                 # (H, 2)
    pel_quat = anchor[:, 3:7]              # (H, 4) wxyz
    yaw = _yaw_from_wxyz(pel_quat)         # (H,)

    # 1) decode action's 4 EE from each-frame nav back to WORLD
    ee_block = action[:, :36].reshape(H, 4, 9)            # (H, 4, 9) ee-major
    ee_pos_nav = ee_block[..., 0:3]                       # (H, 4, 3)
    ee_rot6d = ee_block[..., 3:9]                         # (H, 4, 6)
    R_local = _rot6d_to_mat(ee_rot6d)                    # (H, 4, 3, 3)  nav-frame

    # world from nav: world = R_z(yaw) * nav + pelvis(xy, z=0)
    c, s = torch.cos(yaw), torch.sin(yaw)
    R_w2n = _R2_from_yaw(yaw)              # (H, 2, 2)   world->nav (we need its inverse = transpose)
    R_n2w = R_w2n.transpose(-2, -1)        # (H, 2, 2)   nav->world
    ee_xy_w = torch.einsum("hij,hkj->hki", R_n2w, ee_pos_nav[..., :2]) + pel_xy[:, None, :]
    ee_z_w = ee_pos_nav[..., 2]
    ee_pos_w = torch.cat([ee_xy_w, ee_z_w.unsqueeze(-1)], dim=-1)   # (H, 4, 3)

    # rotation in world: R_world = R_z(yaw) @ R_local
    R_yaw_3d = torch.zeros(H, 3, 3, dtype=action.dtype, device=action.device)
    R_yaw_3d[:, 0, 0] = c; R_yaw_3d[:, 0, 1] = -s
    R_yaw_3d[:, 1, 0] = s; R_yaw_3d[:, 1, 1] = c
    R_yaw_3d[:, 2, 2] = 1.0
    R_world = torch.einsum("hij,hkjm->hkim", R_yaw_3d, R_local)  # (H, 4, 3, 3)

    # 2) project back into chunk[0]'s anchor
    pel_xy0 = pel_xy[0:1]    # (1, 2)
    yaw0 = yaw[0:1]          # (1,)
    R_w2n0 = _R2_from_yaw(yaw0)[0]  # (2, 2)
    c0, s0 = torch.cos(yaw0[0]), torch.sin(yaw0[0])
    R_yaw0_3d = torch.eye(3, dtype=action.dtype, device=action.device)
    R_yaw0_3d[0, 0] = c0; R_yaw0_3d[0, 1] = -s0
    R_yaw0_3d[1, 0] = s0; R_yaw0_3d[1, 1] = c0
    R_yaw0_inv = R_yaw0_3d.transpose(0, 1)

    # pos in nav0
    ee_xy_nav0 = torch.einsum("ij,hkj->hki", R_w2n0, ee_pos_w[..., :2] - pel_xy0[None])
    ee_pos_nav0 = torch.cat([ee_xy_nav0, ee_pos_w[..., 2:3]], dim=-1)
    # rot in nav0
    R_local_nav0 = torch.einsum("ij,hkjm->hkim", R_yaw0_inv, R_world)
    ee_rot6d_nav0 = _mat_to_rot6d(R_local_nav0)

    # write back (EE-major: (H, 4, 9) -> (H, 36))
    new_block = torch.cat([ee_pos_nav0, ee_rot6d_nav0], dim=-1)  # (H, 4, 9)
    new_action = action.clone()
    new_action[:, :36] = new_block.reshape(H, 36)
    item["action"] = new_action
    return item


def project_world_action_to_self_pelvis_nav(item: dict) -> dict:
    """bfm-v1.3 transform: project ref EE world -> per-frame pelvis-nav(t).

    Per-frame action layout (42 dims):
      [ 0:36] ref 4 EE in WORLD, EE-major 9 cols each (pos + rot6d)
      [36:38] l/r foot contact
      [38]    right_hand_close
      [39:42] sim pelvis t->t+1 (dx, dy, dyaw) in sim_nav(t) -- pass through

    After transform only [0:36] changes: EE re-expressed in sim_nav(t) using
    `observation.anchor_pose` (sim pelvis world).  Contact/grip/pelvis-vel
    untouched.
    """
    action = item["action"]                # (H, 42)
    anchor = item["observation.anchor_pose"]  # (H, 7): sim pelvis world [pos, quat wxyz]
    if action.ndim != 2 or action.shape[1] < 36:
        return item
    H = action.shape[0]
    if anchor.shape[0] != H:
        return item

    pel_pos = anchor[:, :3]               # (H, 3)
    pel_quat = anchor[:, 3:7]             # (H, 4) wxyz
    yaw = _yaw_from_wxyz(pel_quat)        # (H,)

    ee_block = action[:, :36].reshape(H, 4, 9)
    ee_pos_w = ee_block[..., 0:3]         # (H, 4, 3)
    ee_rot6d_w = ee_block[..., 3:9]       # (H, 4, 6)
    R_world = _rot6d_to_mat(ee_rot6d_w)   # (H, 4, 3, 3)

    # world -> nav(t):  nav_xy = R_w2n @ (world_xy - pel_xy);  nav_z = world_z
    # (nav is yaw-only on z=0 ground; T_world2nav sets pelvis z to 0).
    R_w2n = _R2_from_yaw(yaw)             # (H, 2, 2)
    ee_xy_nav = torch.einsum("hij,hkj->hki", R_w2n, ee_pos_w[..., :2] - pel_pos[:, None, :2])
    ee_z_nav = ee_pos_w[..., 2:3]
    ee_pos_nav = torch.cat([ee_xy_nav, ee_z_nav], dim=-1)

    # R_local = R_yaw^T @ R_world
    c, s = torch.cos(yaw), torch.sin(yaw)
    R_yaw_3d = torch.zeros(H, 3, 3, dtype=action.dtype, device=action.device)
    R_yaw_3d[:, 0, 0] = c; R_yaw_3d[:, 0, 1] = -s
    R_yaw_3d[:, 1, 0] = s; R_yaw_3d[:, 1, 1] = c
    R_yaw_3d[:, 2, 2] = 1.0
    R_yaw_inv = R_yaw_3d.transpose(-2, -1)
    R_local = torch.einsum("hij,hkjm->hkim", R_yaw_inv, R_world)
    ee_rot6d_local = _mat_to_rot6d(R_local)

    new_block = torch.cat([ee_pos_nav, ee_rot6d_local], dim=-1)  # (H, 4, 9)
    new_action = action.clone()
    new_action[:, :36] = new_block.reshape(H, 36)
    item["action"] = new_action
    return item


def _reproject_ee_block_to_target_writer(
    ee_pos_src: torch.Tensor,    # (..., 4, 3) in nav_src
    ee_rot6d_src: torch.Tensor,  # (..., 4, 6) in nav_src, WRITER layout
    pel_xy_src: torch.Tensor,    # (..., 2)
    yaw_src: torch.Tensor,       # (...,)
    pel_xy_tgt: torch.Tensor,    # (2,)
    yaw_tgt: torch.Tensor,       # scalar
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-express 4-EE pose (pos + rot6d) from per-row nav(src) into a
    SINGLE target nav frame.  Pelvis-nav is yaw-only on the z=0 plane,
    so xy gets a 2D rotation + translation, z is passthrough; rotation
    composes via R_z(yaw_src) on the source side and R_z(yaw_tgt)^T on
    the target side.  Used by both action-chunk reanchor and state-
    history reproject (which is just this with the source axis = rows
    of history and target = the current pelvis-nav).
    """
    # src -> world
    R_n2w_src = _R2_from_yaw(yaw_src).transpose(-2, -1)            # (..., 2, 2)
    ee_xy_w = torch.einsum("...ij,...kj->...ki", R_n2w_src, ee_pos_src[..., :2]) \
              + pel_xy_src[..., None, :]
    ee_pos_w = torch.cat([ee_xy_w, ee_pos_src[..., 2:3]], dim=-1)  # (..., 4, 3)

    c_s, s_s = torch.cos(yaw_src), torch.sin(yaw_src)
    R_yaw_src = torch.zeros(*yaw_src.shape, 3, 3,
                            dtype=ee_pos_src.dtype, device=ee_pos_src.device)
    R_yaw_src[..., 0, 0] = c_s; R_yaw_src[..., 0, 1] = -s_s
    R_yaw_src[..., 1, 0] = s_s; R_yaw_src[..., 1, 1] = c_s
    R_yaw_src[..., 2, 2] = 1.0
    R_local_src = _rot6d_to_mat_writer(ee_rot6d_src)               # (..., 4, 3, 3)
    R_world = torch.einsum("...ij,...kjm->...kim", R_yaw_src, R_local_src)

    # world -> tgt
    R_w2n_tgt = _R2_from_yaw(yaw_tgt)                                # (2, 2)
    ee_xy_tgt = torch.einsum("ij,...kj->...ki",
                              R_w2n_tgt, ee_pos_w[..., :2] - pel_xy_tgt)
    ee_pos_tgt = torch.cat([ee_xy_tgt, ee_pos_w[..., 2:3]], dim=-1)

    c_t, s_t = torch.cos(yaw_tgt), torch.sin(yaw_tgt)
    R_yaw_tgt = torch.eye(3, dtype=ee_pos_src.dtype, device=ee_pos_src.device)
    R_yaw_tgt[0, 0] = c_t; R_yaw_tgt[0, 1] = -s_t
    R_yaw_tgt[1, 0] = s_t; R_yaw_tgt[1, 1] = c_t
    R_local_tgt = torch.einsum("ij,...kjm->...kim", R_yaw_tgt.transpose(0, 1), R_world)
    ee_rot6d_tgt = _mat_to_rot6d_writer(R_local_tgt)

    return ee_pos_tgt, ee_rot6d_tgt


def reanchor_chunk_and_reproject_state_history(item: dict) -> dict:
    """Combined transform:
      * action chunk -> chunk[0]'s pelvis-nav (same as
        reanchor_chunk_to_first_frame).  Anchor for the chunk = the FIRST
        action-aligned anchor row = anchor[-H_action] in the new union
        layout (where state offsets come first).
      * observation.state history -> the SAME current nav.  Each row
        state[i] is written in nav(t + state_offset[i]) at dataset-build
        time; we reproject it to nav(t + 0) using the matching per-row
        anchor and the current anchor.

    Expected layout (factory.py does this automatically when both
    state_delta_indices and action_delta_indices are set on the policy):
      observation.state         : (N_state, 36)
      action                    : (H_action, 39)  EE-major 4*9 + contact 2 + grip 1
      observation.anchor_pose   : (N_state + H_action - 1, 7)
                                  rows 0..N_state-1            -> state history
                                  rows N_state-1..end          -> action chunk
                                  row  N_state-1               -> CURRENT nav

    1-d state (no stacking) and old anchor layout (no state union) are
    both detected and handled (this transform reduces to the action-only
    reanchor in those cases).
    """
    action = item["action"]                  # (H_action, 39)
    state = item["observation.state"]        # (N_state, 36) or (36,)
    anchor = item["observation.anchor_pose"] # (N_anchor, 7)
    if action.ndim != 2 or action.shape[1] < 36 or anchor.ndim != 2:
        return item

    H_action = action.shape[0]
    N_anchor = anchor.shape[0]
    if N_anchor < H_action:
        return item

    # Action anchors are the LAST H_action rows of the union (sorted offsets,
    # action starts at 0 and runs forward).  Current anchor is action[0].
    action_anchor = anchor[-H_action:]
    cur_pel_xy = action_anchor[0, :2]
    cur_yaw = _yaw_from_wxyz(action_anchor[0, 3:7])

    # --- 1) Reanchor action chunk to current nav -------------------------------
    pel_xy_chunk = action_anchor[:, :2]
    yaw_chunk = _yaw_from_wxyz(action_anchor[:, 3:7])
    ee_block = action[:, :36].reshape(H_action, 4, 9)
    ee_pos_cur, ee_rot6d_cur = _reproject_ee_block_to_target_writer(
        ee_block[..., 0:3], ee_block[..., 3:9],
        pel_xy_chunk, yaw_chunk,
        cur_pel_xy, cur_yaw,
    )
    new_action = action.clone()
    new_action[:, :36] = torch.cat([ee_pos_cur, ee_rot6d_cur], dim=-1).reshape(H_action, 36)
    item["action"] = new_action

    # --- 2) Reproject state history to current nav -----------------------------
    if state.ndim == 1:
        return item  # no history stack -> state already at current frame
    N_state = state.shape[0]
    if N_state < 1 or N_anchor < N_state + H_action - 1:
        # Layout doesn't match expected union -> assume anchor wasn't
        # widened to include history offsets; can't reproject reliably.
        return item

    state_anchor = anchor[:N_state]
    pel_xy_state = state_anchor[:, :2]
    yaw_state = _yaw_from_wxyz(state_anchor[:, 3:7])
    # v1.6+: state may carry extra frame-invariant cols (sim foot contact
    # bits) after the 36-dim EE block.  Reanchor only the first 36 cols;
    # passthrough the rest unchanged.
    state_ee_block = state[..., :36].reshape(N_state, 4, 9)
    state_pos_cur, state_rot6d_cur = _reproject_ee_block_to_target_writer(
        state_ee_block[..., 0:3], state_ee_block[..., 3:9],
        pel_xy_state, yaw_state,
        cur_pel_xy, cur_yaw,
    )
    new_ee_36 = torch.cat([state_pos_cur, state_rot6d_cur], dim=-1).reshape(N_state, 36)
    if state.shape[-1] > 36:
        new_state = torch.cat([new_ee_36, state[..., 36:]], dim=-1)
    else:
        new_state = new_ee_36
    item["observation.state"] = new_state

    return item


def reanchor_chunk_keep_state_local(item: dict) -> dict:
    """bfm-v1.7 transform: action chunk reanchored to chunk[0]'s pelvis-nav
    (same as `reanchor_chunk_and_reproject_state_history`); state history
    LEFT UNCHANGED (each row stays in its own nav(t-k) frame).

    State pass-through means:
      * deploy doesn't need a global-anchor stack to reproject history --
        each tick's state is computed in the current nav and stacked raw
      * model sees per-frame ego-relative EE, plus the 2 contact bits in
        cols 36:38 (= sim foot stance at that frame)
      * post-transform stats over state look like raw per-row stats
        (no chunk-anchor drift) so std is naturally tight without recompute
    """
    action = item["action"]
    anchor = item["observation.anchor_pose"]
    if action.ndim != 2 or action.shape[1] < 36 or anchor.ndim != 2:
        return item

    H_action = action.shape[0]
    if anchor.shape[0] < H_action:
        return item

    # Same action-chunk anchor extraction as the v1.6 transform.
    action_anchor = anchor[-H_action:]
    cur_pel_xy = action_anchor[0, :2]
    cur_yaw = _yaw_from_wxyz(action_anchor[0, 3:7])

    pel_xy_chunk = action_anchor[:, :2]
    yaw_chunk = _yaw_from_wxyz(action_anchor[:, 3:7])
    ee_block = action[:, :36].reshape(H_action, 4, 9)
    ee_pos_cur, ee_rot6d_cur = _reproject_ee_block_to_target_writer(
        ee_block[..., 0:3], ee_block[..., 3:9],
        pel_xy_chunk, yaw_chunk,
        cur_pel_xy, cur_yaw,
    )
    new_action = action.clone()
    new_action[:, :36] = torch.cat([ee_pos_cur, ee_rot6d_cur], dim=-1).reshape(H_action, 36)
    item["action"] = new_action
    # observation.state passes through unmodified.
    return item


def reanchor_chunk_keep_state_local_state_dropout(
    item: dict,
    state_dropout_p: float = 0.0,
) -> dict:
    """v1.7 transform + train-time state dropout.

    Combats inference-time covariate shift: at deploy the BFM-tracking-VLA-
    chunks state distribution drifts off the BFM-tracking-BVH training
    distribution, and the model has zero robustness to mildly-OOD state
    values (verified via state ablations: real state and last-frame-repeat
    both fail at the same rate; zero state succeeds).  Forcing the model
    to occasionally make predictions WITHOUT any state token trains an
    image+task fallback pathway that the deploy-time policy can lean on.

    Configure via DatasetConfig.item_transform_kwargs, e.g.
        item_transform_path: "chunk_transforms:reanchor_chunk_keep_state_local_state_dropout"
        item_transform_kwargs: {state_dropout_p: 0.3}
    factory.py binds these kwargs to the TRAIN dataset only; the test
    dataset falls back to the default (state_dropout_p=0.0 = identical to
    `reanchor_chunk_keep_state_local`).  Deploy (play_vla_infer.py)
    bypasses LeRobotDataset entirely and never reaches this function, so
    inference is unaffected regardless of config.
    """
    item = reanchor_chunk_keep_state_local(item)
    if state_dropout_p > 0.0 and torch.rand(()).item() < state_dropout_p:
        item["observation.state"] = torch.zeros_like(item["observation.state"])
    return item
