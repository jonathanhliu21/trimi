"""Mirror the real TriMI hand onto its MuJoCo model, joint for joint.

    python3 scripts/sim/trimi_mirror.py              # move the hand, the model follows
    python3 scripts/sim/trimi_mirror.py --stops      # also check the joints' homing stops

The motors stay torque-disabled. Joint angles are written straight into the
model (no physics). Build the model first with
scripts/sim/build_trimi_sim_model.py; the motor-to-joint map is
config/trimi_v2_joint_map.yaml, whose zeros scripts/sim/sync_hard_stops.py sets
from the homing hard stops.

With --stops, push each joint by hand firmly into the stop homing drives it
to; on exit the mirror reports, per joint, how far past (+) or short (-) of the
model's stop the model is when the real joint is at its stop.
--save-stops then also moves the zeros so the model meets its stop exactly
there.

By default the hand calibration is the one 'trimi-live home' saves
(~/.config/trimi-copilot/hand.yaml), and the port is the only adapter in
/dev/serial/by-id.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

# Support both direct execution and package imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from data_collection.collection.config import DEFAULT_HAND_CONFIG

DEFAULT_MODEL = Path("assets/trimi_v2/trimi_v2.xml")
DEFAULT_JOINT_MAP = Path("config/trimi_v2_joint_map.yaml")
# Where 'trimi-live home' (TriMI-Copilot) saves the hand's current offsets.
HOMED_HAND_CONFIG = Path.home() / ".config/trimi-copilot/hand.yaml"
SERIAL_BY_ID = Path("/dev/serial/by-id")
DEFAULT_RATE_HZ = 60.0
STOP_REACHED_RAD = math.radians(10.0)  # a joint left farther from its stop was not pushed to it


@dataclass(frozen=True)
class JointMapEntry:
    motor_id: int
    urdf_joint: str
    label: str
    sign: float
    scale: float
    zero: float  # leader reading when the URDF joint is at 0


@dataclass(frozen=True)
class HomingStop:
    reading: float  # calibrated reading at the stop
    direction: int  # sign of the reading's motion towards the stop


def homing_rows(hand: Mapping) -> list:
    homing = hand.get("homing") or {}
    return list(homing.get("thumb", [])) + list(homing.get("fingers", []))


def homing_stops(hand: Mapping) -> dict[int, HomingStop]:
    """Reading at each motor's homing stop: home_offset = raw_stop + cad_offset."""

    signs = {int(k): float(v) for k, v in hand["joint_signs"].items()}
    stops = {}
    for motor_id, _label, cad, direction in homing_rows(hand):
        sign = signs[int(motor_id)]
        stops[int(motor_id)] = HomingStop(-sign * float(cad), int(np.sign(sign * direction)))
    return stops


def parse_joint_map(payload: Mapping) -> tuple[JointMapEntry, ...]:
    joints = payload.get("joints") if isinstance(payload, Mapping) else None
    if not isinstance(joints, Mapping) or not joints:
        raise ValueError("Joint map needs a non-empty 'joints' mapping")
    entries = []
    for motor_id, item in joints.items():
        if not isinstance(item, Mapping) or "urdf_joint" not in item:
            raise ValueError(f"Motor {motor_id}: expected a mapping with urdf_joint")
        if "offset" in item:
            raise ValueError(
                f"Motor {motor_id}: 'offset' is no longer used; give 'zero', the "
                "leader reading at the URDF zero (scripts/sim/sync_hard_stops.py --write sets it)"
            )
        entry = JointMapEntry(
            motor_id=int(motor_id),
            urdf_joint=str(item["urdf_joint"]),
            label=str(item.get("label", f"motor {motor_id}")),
            sign=float(item.get("sign", 1.0)),
            scale=float(item.get("scale", 1.0)),
            zero=float(item.get("zero", 0.0)),
        )
        if entry.sign not in (1.0, -1.0):
            raise ValueError(f"Motor {motor_id}: sign must be 1 or -1")
        if not (math.isfinite(entry.scale) and entry.scale > 0):
            raise ValueError(f"Motor {motor_id}: scale must be finite and positive")
        if not math.isfinite(entry.zero):
            raise ValueError(f"Motor {motor_id}: zero must be finite")
        entries.append(entry)
    joints_used = [entry.urdf_joint for entry in entries]
    duplicates = sorted({name for name in joints_used if joints_used.count(name) > 1})
    if duplicates:
        raise ValueError(f"URDF joints mapped more than once: {duplicates}")
    return tuple(entries)


def load_joint_map(path: Path) -> tuple[JointMapEntry, ...]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"Joint map not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return parse_joint_map(yaml.safe_load(stream))


def write_zeros(path: Path, zeros: Mapping[int, float]) -> None:
    """Replace the 'zero' value on each motor's line, keeping the rest of the file."""

    import yaml

    text = path.read_text(encoding="utf-8")
    missing = []
    for motor_id, value in zeros.items():
        pattern = re.compile(
            rf"^(\s*{int(motor_id)}\s*:\s*\{{[^}}\n]*?\bzero\s*:\s*)([^,}}\s]+)", re.M
        )
        text, count = pattern.subn(lambda match: f"{match.group(1)}{value:.4f}", text, count=1)
        if not count:
            missing.append(int(motor_id))
    if missing:
        values = ", ".join(f"{m}: {zeros[m]:.4f}" for m in missing)
        raise ValueError(
            f"Could not find a one-line entry with 'zero:' for motors {missing} in {path}; "
            f"set them by hand: {values}"
        )
    parse_joint_map(yaml.safe_load(text))
    path.write_text(text, encoding="utf-8")


class HandMirror:
    """Write leader readings (in hand motor order) into the model's joints."""

    def __init__(self, model, entries: Sequence[JointMapEntry], motor_ids: Sequence[int]):
        import mujoco

        by_motor = {entry.motor_id: entry for entry in entries}
        missing = [int(m) for m in motor_ids if int(m) not in by_motor]
        if missing:
            raise ValueError(f"Joint map has no entry for motor IDs {missing}")
        self.entries = tuple(by_motor[int(m)] for m in motor_ids)
        addresses = []
        for entry in self.entries:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry.urdf_joint)
            if joint < 0:
                raise ValueError(
                    f"Motor {entry.motor_id} ({entry.label}): no joint "
                    f"{entry.urdf_joint!r} in the model"
                )
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Joint {entry.urdf_joint!r} is not a hinge")
            addresses.append(int(model.jnt_qposadr[joint]))
        self.qpos_addresses = np.asarray(addresses, dtype=np.int64)
        self.gain = np.asarray([e.sign * e.scale for e in self.entries], dtype=np.float64)
        self.zero = np.asarray([e.zero for e in self.entries], dtype=np.float64)

    def to_urdf(self, positions: Sequence[float]) -> np.ndarray:
        values = np.asarray(positions, dtype=np.float64)
        if values.shape != self.gain.shape:
            raise ValueError(
                f"Expected {self.gain.size} hand positions, got shape {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("Hand positions must be finite")
        return self.gain * (values - self.zero)

    def apply(self, data, positions: Sequence[float]) -> np.ndarray:
        urdf = self.to_urdf(positions)
        data.qpos[self.qpos_addresses] = urdf
        return urdf


def describe(mirror: HandMirror) -> str:
    rows = [f"{'motor':>5}  {'label':15s} {'URDF joint':13s} mapping"]
    for entry in mirror.entries:
        gain = f"{'-' if entry.sign < 0 else ''}{entry.scale:.4g}"
        rows.append(
            f"{entry.motor_id:>5}  {entry.label:15s} {entry.urdf_joint:13s} "
            f"q = {gain} * (leader - {entry.zero:+.3f})"
        )
    return "\n".join(rows)


class StopCheck:
    """How far each joint gets pushed towards the stop homing drives it into."""

    def __init__(self, mirror: HandMirror, stops: Mapping[int, HomingStop]):
        missing = [entry.motor_id for entry in mirror.entries if entry.motor_id not in stops]
        if missing:
            raise ValueError(f"The hand's homing table has no row for motors {missing}")
        self.entries = mirror.entries
        self.reading = np.asarray([stops[e.motor_id].reading for e in self.entries], dtype=np.float64)
        self.direction = np.asarray([stops[e.motor_id].direction for e in self.entries], dtype=np.float64)
        self.farthest = np.full(len(self.entries), -np.inf)  # motor rad past the homing stop reading

    def update(self, positions: Sequence[float]) -> None:
        past = self.direction * (np.asarray(positions, dtype=np.float64) - self.reading)
        np.maximum(self.farthest, past, out=self.farthest)

    def reached(self) -> list[int]:
        return [i for i, past in enumerate(self.farthest) if past > -STOP_REACHED_RAD]

    def zeros(self) -> dict[int, float]:
        """Zeros that put each hand-pushed stop on the model's stop (where homing's stop maps now)."""

        return {self.entries[i].motor_id: self.entries[i].zero + self.direction[i] * self.farthest[i]
                for i in self.reached()}

    def report(self) -> str:
        reached = set(self.reached())
        rows = [f"{'motor':>5}  {'label':15s} {'homing stop':>11} {'pushed to':>9} {'model at the real stop':>24}"]
        for i, entry in enumerate(self.entries):
            if i not in reached:
                rows.append(f"{entry.motor_id:>5}  {entry.label:15s} {self.reading[i]:+11.3f}  not pushed to its stop")
                continue
            pushed = self.reading[i] + self.direction[i] * self.farthest[i]
            model = math.degrees(entry.scale * self.farthest[i])
            where = "past" if model > 0 else "short of"
            rows.append(f"{entry.motor_id:>5}  {entry.label:15s} {self.reading[i]:+11.3f} {pushed:+9.3f}"
                        f" {abs(model):8.1f} deg {where} its stop")
        return "\n".join(rows)


def mirror_loop(
    hand,
    model,
    data,
    mirror: HandMirror,
    viewer,
    *,
    rate_hz: float,
    print_period_s: float | None = None,
    on_positions: Callable[[np.ndarray], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Copy hand readings into the viewer until it closes; return frames shown."""

    import mujoco

    period_s = 1.0 / rate_hz
    next_frame = clock()
    next_print = next_frame
    frames = 0
    while viewer.is_running():
        positions = np.asarray(hand.read_pos(), dtype=np.float64)
        if on_positions is not None:
            on_positions(positions)
        with viewer.lock():
            mirror.apply(data, positions)
            mujoco.mj_kinematics(model, data)
        viewer.sync()
        frames += 1
        now = clock()
        if print_period_s is not None and now >= next_print:
            print("leader " + " ".join(
                f"{entry.motor_id}:{value:+.3f}"
                for entry, value in zip(mirror.entries, positions)
            ), flush=True)
            next_print = now + print_period_s
        next_frame += period_s
        if next_frame < now:
            next_frame = now
        sleep(max(0.0, next_frame - now))
    return frames


def default_hand_config() -> Path:
    return HOMED_HAND_CONFIG if HOMED_HAND_CONFIG.is_file() else DEFAULT_HAND_CONFIG


def find_hand_port(directory: Path = SERIAL_BY_ID) -> str:
    ports = sorted(directory.iterdir()) if directory.is_dir() else []
    if len(ports) != 1:
        found = ", ".join(p.name for p in ports) or "none"
        raise RuntimeError(f"Pass --hand-port: expected one adapter in {directory}, found {found}")
    return str(ports[0])


def run(args: argparse.Namespace) -> int:
    import mujoco
    import yaml
    import mujoco.viewer
    from midas_hand_api import MidasHand

    from data_collection.collection.hand_common import load_hand_config, verify_connected_hand

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Build it with "
            "python3 scripts/sim/build_trimi_sim_model.py <TRIMI_URDF zip>"
        )
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    port = args.hand_port or find_hand_port()
    hand_config = load_hand_config(args.hand_config, port)
    mirror = HandMirror(model, load_joint_map(args.joint_map), hand_config.motor_ids)
    check = None
    if args.stops:
        with args.hand_config.open("r", encoding="utf-8") as stream:
            check = StopCheck(mirror, homing_stops(yaml.safe_load(stream)))
    print(f"Hand calibration: {args.hand_config}\n{describe(mirror)}")

    hand = MidasHand(config=hand_config, autoconnect=True)
    try:
        verify_connected_hand(hand, role="TriMI")
        hand.disable_torque()
        print(f"TriMI connected on {hand.port}; torque off.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = (0.0, -0.03, -0.06)
                viewer.cam.distance = 0.45
                viewer.cam.azimuth = -120.0
                viewer.cam.elevation = -25.0
            if check is not None:
                print("Push each joint by hand firmly into the stop homing drives it to.")
            print("Mirroring. Close the viewer or press Ctrl-C to stop.")
            mirror_loop(
                hand, model, data, mirror, viewer,
                rate_hz=args.rate_hz,
                print_period_s=0.5 if args.print else None,
                on_positions=None if check is None else check.update,
            )
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        hand.shutdown()
    if check is not None:
        print(f"\nHoming stops, pushed by hand (readings in motor radians):\n{check.report()}")
        if args.save_stops and check.zeros():
            write_zeros(args.joint_map, check.zeros())
            print(f"Moved the zeros of motors {sorted(check.zeros())} in {args.joint_map} "
                  "so the model meets its stops there.")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hand-config", type=Path, default=None,
                        help=f"hand calibration YAML (default: {HOMED_HAND_CONFIG} if it exists, "
                             f"else {DEFAULT_HAND_CONFIG})")
    parser.add_argument("--hand-port", help=f"serial device (default: the one adapter in {SERIAL_BY_ID})")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                        help="MJCF from build_trimi_sim_model.py (default: %(default)s)")
    parser.add_argument("--joint-map", type=Path, default=DEFAULT_JOINT_MAP,
                        help="motor -> URDF joint map (default: %(default)s)")
    parser.add_argument("--stops", action="store_true",
                        help="on exit, report where each joint pushed by hand into its homing stop "
                             "puts the model relative to the model's stop")
    parser.add_argument("--save-stops", action="store_true",
                        help="with --stops: move the zeros of the joints pushed to their stop so the "
                             "model meets its stop there")
    parser.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ,
                        help="read and redraw rate (default: %(default)s)")
    parser.add_argument("--print", action="store_true",
                        help="print the calibrated motor readings twice a second")
    args = parser.parse_args(argv)
    if not (math.isfinite(args.rate_hz) and args.rate_hz > 0):
        parser.error("--rate-hz must be finite and positive")
    args.stops = args.stops or args.save_stops
    args.hand_config = (args.hand_config or default_hand_config()).expanduser()
    args.model = args.model.expanduser()
    args.joint_map = args.joint_map.expanduser()
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
