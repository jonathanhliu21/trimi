import numpy as np
from scipy.spatial.transform import Rotation as R


def vec6dof_to_homogeneous_matrix(vec6: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, rx, ry, rz] with rotvec rotation to a 4x4 transform."""
    T = np.eye(4)
    T[:3, 3] = vec6[:3]
    T[:3, :3] = R.from_rotvec(vec6[3:]).as_matrix()
    return T


def invert_transformation(T: np.ndarray) -> np.ndarray:
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


def homogeneous_matrix_to_6dof(T: np.ndarray) -> np.ndarray:
    translation = T[:3, 3]
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate((translation, rotvec))


def absolute_to_relative(abs_actions: np.ndarray) -> np.ndarray:
    """
    Convert [T, 6 + H] absolute actions to a window-relative trajectory.

    The first 6 dims are EEF pose. Remaining dims are hand joints. EEF uses
    SE(3) relative pose from the first timestep; hand uses simple subtraction
    from the first timestep.
    """
    if abs_actions.ndim != 2:
        raise ValueError(f"Expected [T, D] actions, got shape {abs_actions.shape}.")
    if abs_actions.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 action dims ([eef_6, hand]), got {abs_actions.shape[1]}."
        )

    eef_abs = abs_actions[:, :6]
    hand_abs = abs_actions[:, 6:]

    T0 = vec6dof_to_homogeneous_matrix(eef_abs[0])
    T0_inv = invert_transformation(T0)
    eef_rel = np.zeros_like(eef_abs)
    for i in range(abs_actions.shape[0]):
        Tt = vec6dof_to_homogeneous_matrix(eef_abs[i])
        eef_rel[i] = homogeneous_matrix_to_6dof(T0_inv @ Tt)

    hand_rel = hand_abs - hand_abs[0:1]
    return np.concatenate([eef_rel, hand_rel], axis=-1).astype(np.float32)


def relative_to_absolute(rel_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
    """
    Convert [T, 6 + H] relative actions back to absolute actions.

    current_state must be [6 + H] absolute current EEF pose and hand state.
    """
    if rel_actions.ndim != 2:
        raise ValueError(f"Expected [T, D] relative actions, got {rel_actions.shape}.")
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    if rel_actions.shape[1] != current_state.shape[0]:
        raise ValueError(
            "rel_actions and current_state dimension mismatch: "
            f"{rel_actions.shape[1]} vs {current_state.shape[0]}."
        )

    eef_rel = rel_actions[:, :6]
    hand_rel = rel_actions[:, 6:]
    current_eef = current_state[:6]
    current_hand = current_state[6:]

    T_current = vec6dof_to_homogeneous_matrix(current_eef)
    eef_abs = np.zeros_like(eef_rel)
    for i in range(rel_actions.shape[0]):
        T_rel = vec6dof_to_homogeneous_matrix(eef_rel[i])
        eef_abs[i] = homogeneous_matrix_to_6dof(T_current @ T_rel)

    hand_abs = hand_rel + current_hand
    return np.concatenate([eef_abs, hand_abs], axis=-1).astype(np.float32)
