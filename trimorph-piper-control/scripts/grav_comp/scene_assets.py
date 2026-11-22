"""The Piper and the TRIMI hand for the gravity-comp checks, and the old rig's cart and post.

Taken from trimorph-piper-control's workspace sim (scripts/workspace/analysis.py) so this
folder runs on its own. World axes: x east, y north, z up, floor at z = 0.

assets/piper_description.urdf, assets/meshes/: the official AgileX Piper model (vendored from
midas-piper-control); its link6 frame reproduces the SDK FK to 0.1 mm.
assets/trimi_hand.npz: convex hulls of the TRIMI hand, Quest controller, camera and flange
adapter from the Onshape ARM.step, in link6 axes, turned so J6 = 0 is palm down.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

ASSET_DIR = Path(__file__).resolve().parent / "assets"
HAND_ASSET = ASSET_DIR / "trimi_hand.npz"
ARM_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6")

# The old rig, from ARM.step (x east, y north): an 8020 cart 0.43 m square on casters and a
# 33 cm post at its NE corner, still standing in the config-2 layout. Centre x, y, z and half
# sizes x, y, z (m).
CART_BOXES = ((-0.195, -0.141, 0.300, 0.215, 0.215, 0.300),   # cart frame (treated as solid)
              (-0.0055, 0.049, 0.765, 0.0255, 0.025, 0.165))  # post


def load_hand(path: Path = HAND_ASSET) -> dict[str, np.ndarray]:
    """Hull point sets of the hand in link6 axes, palm toward link6 -y."""
    with np.load(path) as data:
        return {k[5:]: data[k] for k in data.files if k.startswith("hull_")}


def urdf_text(rotation: np.ndarray, position: np.ndarray) -> str:
    """The Piper URDF with its base at `position` and `rotation` in the world; meshes are
    referenced by file name (set the MjSpec meshdir to ASSET_DIR / "meshes")."""
    text = (ASSET_DIR / "piper_description.urdf").read_text()
    text = re.sub(r"<visual>.*?</visual>", "", text, flags=re.S)  # .dae visuals; .stl collisions double as visuals
    text = re.sub(r'package://[^"]*/', "", text)
    rpy = Rotation.from_matrix(rotation).as_euler("xyz")
    origin = (f'<origin xyz="{position[0]} {position[1]} {position[2]}" '
              f'rpy="{rpy[0]:.12f} {rpy[1]:.12f} {rpy[2]:.12f}"/>')
    text, count = re.subn(r'(<joint name="world_to_base_link".*?)<origin[^>]*/>',
                          lambda match: match.group(1) + origin, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("URDF has no world_to_base_link origin to place the base")
    return text


def quat(rot: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w, x, y, z) of a rotation matrix."""
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(rot, float).reshape(-1))
    return out
