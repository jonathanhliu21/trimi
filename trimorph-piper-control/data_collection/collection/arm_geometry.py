"""Pure Piper link frames from the executor's SDK modified-DH model.

This imports kinematic constants only. It creates no robot, connection, or
motion command; joint limits are deliberately not applied to measured angles.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation


@lru_cache(maxsize=1)
def _piper_mdh():
    from pyAgxArm.utiles.mdh_kinematics import get_mdh

    return get_mdh("piper")


def piper_frames(joints) -> np.ndarray:
    """Return base plus six link transforms, shape (7, 4, 4), metres/radians.

    Uses the same SDK table and ``Rx(alpha) Tx(a) Rz(q+offset) Tz(d)``
    convention as ``replay_actions.piper_model()`` / executor JointController.
    The last frame is the flange, without a guessed hand or tool offset.
    """
    q = np.asarray(joints, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Piper geometry needs six finite joint angles in radians")
    frames = [np.eye(4)]
    for angle, (d, a, alpha, offset) in zip(q, _piper_mdh()):
        ca, sa = math.cos(alpha), math.sin(alpha)
        ct, st = math.cos(angle + offset), math.sin(angle + offset)
        link = np.array([[ct, -st, 0, a],
                         [ca * st, ca * ct, -sa, -sa * d],
                         [sa * st, sa * ct, ca, ca * d],
                         [0, 0, 0, 1]])
        frames.append(frames[-1] @ link)
    return np.stack(frames)


def gravity_up_rotation(gravity_vector) -> np.ndarray:
    """Map Piper base vectors into a view frame with physical up along +Z.

    Keep displayed +X aligned with the horizontal projection of base +X;
    if gravity is parallel to that axis, use base +Y for a stable basis.
    """
    gravity = np.asarray(gravity_vector, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-9:
        raise ValueError("Gravity view needs a finite nonzero three-vector")
    up = -gravity / np.linalg.norm(gravity)
    horizontal = np.eye(3)[0] - up[0] * up
    if np.linalg.norm(horizontal) < 1e-6:
        horizontal = np.eye(3)[1] - up[1] * up
    horizontal /= np.linalg.norm(horizontal)
    return np.stack((horizontal, np.cross(up, horizontal), up))


def flange_pose_error(joints_rad, target_joints_rad) -> tuple[float, float]:
    """Euclidean flange distance (m) and total shortest rotation error (deg).

    The SDK flange frame is also used by the executor. No unmeasured tool or
    hand offset is added. SO(3) magnitude handles Euler wrap and combines all
    rotation axes into one angle in [0, 180] degrees.
    """
    current = piper_frames(joints_rad)[-1]
    target = piper_frames(target_joints_rad)[-1]
    distance_m = np.linalg.norm(current[:3, 3] - target[:3, 3])
    relative = target[:3, :3].T @ current[:3, :3]
    rotation_deg = np.degrees(Rotation.from_matrix(relative).magnitude())
    return float(distance_m), float(rotation_deg)
