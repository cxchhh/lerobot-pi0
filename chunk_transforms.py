"""LeRobot item_transform hooks for chunk re-anchoring.

Drop into LeRobot training via DatasetConfig:
    item_transform_path: "chunk_transforms:reanchor_chunk_to_first_frame"

Assumes the dataset has:
  - `action`               shape (H, 39): 12 pos + 24 rot6d + l_contact + r_contact + grip
                           expressed in each frame's own pelvis-nav frame
  - `observation.anchor_pose` shape (H, 7): per-frame sim pelvis world pose
                           [x, y, z, qw, qx, qy, qz]

After transform: `action` is rewritten so every chunk frame is expressed in
the FIRST frame's pelvis-nav anchor (yaw-only nav frame, anchored at pelvis
xy+yaw at chunk[0]).  Contact / grip channels untouched.
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
    anchor = item["observation.anchor_pose"]  # (H, 7): pelvis world [pos, quat wxyz]
    if action.ndim != 2 or action.shape[1] < 36:
        return item
    H = action.shape[0]
    if anchor.shape[0] != H:
        return item

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
