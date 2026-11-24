#!/usr/bin/env python3
"""Launch the unchanged drag controller and receive-only collector in tmux."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_collection.collection.config import (
    DEFAULT_GRAV_COMP_CALIB, KINESTHETIC_HAND_CONFIG, KINESTHETIC_HAND_PORT,
    PIPER_START_JOINTS_DEG,
)
from scripts.grav_comp.check_start_pose import find_midas_repo

READY = "assist at target - drag away."
COLLECTION_J5_KD_MIT = 0.005
STOP_MESSAGES = ("freeze requested", "SIGINT -> freeze", "quit requested", "session done",
                 "-> freeze+stop", "control loop aborted", "*** DAMP BRAKE")


def drag_ready(text):
    """A freeze is latched for this session, even if a later ready line appears."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).replace("\r", "")
    index = text.rfind(READY + "\n")
    return index >= 0 and not any(word in text for word in STOP_MESSAGES)


def running_drag(channel):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = entry.joinpath("cmdline").read_bytes().decode().strip("\0").split("\0")
        except (OSError, UnicodeError):
            continue
        if "gravity_comp.drag_mode" not in args and not any(a.endswith("/drag_mode.py") for a in args):
            continue
        used_channel = "can0"
        for i, arg in enumerate(args):
            if arg == "--channel" and i + 1 < len(args):
                used_channel = args[i + 1]
            elif arg.startswith("--channel="):
                used_channel = arg.split("=", 1)[1]
        if used_channel == channel:
            found.append(int(entry.name))
    return found


def tmux(*args):
    return subprocess.check_output(["tmux", *map(str, args)], text=True).strip()


def pane_alive(pane):
    try:
        return tmux("display-message", "-p", "-t", pane, "#{pane_dead}") == "0"
    except subprocess.CalledProcessError:
        return False


def collect(session_dir):
    manifest = json.loads((session_dir / "session.json").read_text())
    pane = manifest["drag_pane"]
    log = session_dir / "drag.log"
    print("Waiting for drag mode's 'assist at target' message. Do the support/arming ritual in the left pane.", flush=True)
    while pane_alive(pane):
        if log.exists():
            output = log.read_text(errors="replace")
            if any(word in output for word in STOP_MESSAGES):
                raise RuntimeError("Drag froze or stopped before collection; follow its support/disable prompt")
            if drag_ready(output):
                break
        time.sleep(.2)
    else:
        raise RuntimeError("Drag pane exited before assist reached target; collector was not started")
    if manifest.get("episode_role") == "start_definition":
        print("Drag assist reached target. No task start is saved: position the arm and hand, "
              "then record a short pedal clip for offline start definition.", flush=True)
    else:
        print("Drag assist reached target. Start guide is display-only; use the pedal after reaching the start.", flush=True)
    child = None
    old_handler = signal.getsignal(signal.SIGINT)

    def stop_collector(_signum, _frame):
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop_collector)
    try:
        # Its own process group lets this supervisor forward Ctrl-C exactly once.
        child = subprocess.Popen(manifest["collector_command"], cwd=ROOT, start_new_session=True)
        stop_sent = False
        while child.poll() is None:
            if not stop_sent and (not pane_alive(pane) or not drag_ready(log.read_text(errors="replace"))):
                print("Drag froze or exited; stopping the collector to save the episode.", flush=True)
                stop_collector(None, None)
                stop_sent = True
            time.sleep(.2)
        return child.returncode
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            child.wait()
        print("Collector stopped. In the drag pane press q, support the arm, then Enter to disable. Do not kill the tmux session while torque is live.", flush=True)


def commands(args, midas, session_dir, session_id):
    py = midas / ".venv/bin/python"
    if not py.is_file():
        py = Path(sys.executable)
    snapshot = session_dir / "calibration.json"
    drag = [str(py), "-u", "-m", "gravity_comp.drag_mode", "--force", "--allow-underpowered",
            "--channel", args.channel, "--calib", str(snapshot),
            "--j5-kd", str(getattr(args, "j5_kd", COLLECTION_J5_KD_MIT))]
    if args.disable_j6:
        drag.append("--disable-j6")
    elif args.passive_j6:
        drag.append("--passive-j6")
    elif getattr(args, "passive_j5_j6", False):
        drag.append("--passive-j5-j6")
    collector = [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "data_collection/collect_hand_only.py"),
                 "--arm-mode", "gravity-comp", "--hand-config", str(args.hand_config.resolve()),
                 "--hand-port", args.hand_port, "--dataset-root", str(args.dataset_root.resolve()),
                 "--piper-channel", args.channel, "--grav-comp-calib", str(snapshot),
                 "--drag-session-id", session_id]
    if args.start_profile is not None:
        collector += ["--start-profile", str(session_dir / "task_start.json")]
    elif args.start_home is not None:
        collector += ["--kinesthetic-start", args.start_home]
    else:
        collector.append("--define-start")
    if args.disable_j6:
        collector.append("--grav-comp-disable-j6")
    elif args.passive_j6:
        collector.append("--grav-comp-passive-j6")
    elif getattr(args, "passive_j5_j6", False):
        collector.append("--grav-comp-passive-j5-j6")
    if args.no_camera:
        collector.append("--no-camera")
    if args.no_monitor:
        collector += ["--no-monitor", "--no-piper-joint-monitor"]
    if args.episode_s is not None:
        collector += ["--episode-s", str(args.episode_s)]
    return drag, collector


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "collect":
        return collect(Path(argv[1]))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("data_trimorph_kinesthetic"))
    p.add_argument("--midas-repo", type=Path)
    p.add_argument("--calib", type=Path, default=DEFAULT_GRAV_COMP_CALIB)
    p.add_argument("--channel", default="can0")
    p.add_argument("--hand-config", type=Path, default=ROOT / KINESTHETIC_HAND_CONFIG)
    p.add_argument("--hand-port", default=KINESTHETIC_HAND_PORT)
    start = p.add_mutually_exclusive_group()
    start.add_argument("--start-profile", type=Path, help="saved task start JSON; copied into the session")
    start.add_argument("--define-start", action="store_true",
                       help="record a task-start definition clip without a saved target (default)")
    start.add_argument("--start-home", choices=sorted(PIPER_START_JOINTS_DEG),
                       help="explicit legacy named start")
    p.add_argument("--episode-s", type=float, help="single timed episode; otherwise foot-pedal collection")
    p.add_argument("--j5-kd", type=float, default=COLLECTION_J5_KD_MIT,
                   help="J5 MIT damping during ramp-in/drag (default: %(default)s; 0 requests zero damping)")
    p.add_argument("--no-camera", action="store_true")
    wrist = p.add_mutually_exclusive_group()
    wrist.add_argument("--disable-j6", action="store_true", help="keep the wrist motor off; support it even during freeze-hold")
    wrist.add_argument("--passive-j6", action="store_true",
                       help="keep all six drives electrically enabled with zero J6 MIT gains/torque; "
                            "support the wrist even during braking and freeze-hold")
    wrist.add_argument("--passive-j5-j6", action="store_true",
                       help="keep all six drives electrically enabled with zero J5/J6 MIT gains/torque; "
                            "support the wrist even during braking and freeze-hold")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print commands only; no hardware, tmux, or files")
    p.add_argument("--preflight-only", action="store_true", help="check dependencies, CAN, tests and start torque, then exit")
    p.add_argument("--detach", action="store_true", help="create panes and print attach command")
    args = p.parse_args(argv)
    if args.start_profile is None and args.start_home is None:
        args.define_start = True
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.channel):
        p.error("invalid CAN interface name")
    if args.episode_s is not None and (not 0 < args.episode_s < float("inf")):
        p.error("--episode-s must be finite and positive")
    if not 0.0 <= args.j5_kd <= 5.0:
        p.error("--j5-kd must be finite and within [0, 5]")
    midas = (args.midas_repo or find_midas_repo()).expanduser().resolve()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.hand_config = args.hand_config.expanduser().resolve()
    profile = profile_bytes = profile_hash = None
    if args.start_profile is not None:
        from data_collection.collection.task_start import validate_task_start
        args.start_profile = args.start_profile.expanduser().resolve()
        profile_bytes = args.start_profile.read_bytes()
        profile = validate_task_start(json.loads(profile_bytes))
        profile_hash = hashlib.sha256(profile_bytes).hexdigest()
    session_dir = args.dataset_root / "sessions" / session_id
    drag, collector = commands(args, midas, session_dir, session_id)
    if args.dry_run:
        print("Drag (cwd " + str(midas) + "): " + shlex.join(drag))
        print("Collector (after assist at target): " + shlex.join(collector))
        return 0
    # Keep competing launches from passing the process check simultaneously.
    with open(f"/tmp/trimi-kinesthetic-{args.channel}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if running_drag(args.channel):
            raise RuntimeError(f"A drag controller already runs on {args.channel}; stop it with q/support/Enter first")
        for binary in ("tmux", "ip"):
            if not shutil.which(binary):
                raise RuntimeError(f"Missing executable: {binary}")
        link = json.loads(subprocess.check_output(["ip", "-json", "link", "show", args.channel], text=True))
        if not link or "UP" not in link[0].get("flags", []):
            raise RuntimeError(f"{args.channel} is not up; run setup_arm_can.sh first")
        if not Path(args.hand_port).exists():
            raise RuntimeError(f"Hand port missing: {args.hand_port}")
        from data_collection.collection.hand_common import load_hand_config
        load_hand_config(args.hand_config, args.hand_port)
        calibration = args.calib.expanduser().read_bytes()
        help_text = subprocess.check_output([drag[0], "-m", "gravity_comp.drag_mode", "--help"], cwd=midas, text=True)
        required_options = ["--j5-kd"]
        if args.passive_j5_j6:
            required_options.append("--passive-j5-j6")
        elif args.disable_j6 or args.passive_j6:
            required_options.append("--disable-j6" if args.disable_j6 else "--passive-j6")
        for option in required_options:
            if option not in help_text:
                raise RuntimeError(f"This midas gravity_comp checkout does not support {option}; update it first")
        # Validate the exact bytes later copied into the session, never a racing source file.
        torque = {"ok": None, "status": "not_checked",
                  "reason": "task start is not defined; definition recording"}
        if not args.define_start:
            with tempfile.TemporaryDirectory(prefix="kinesthetic-preflight-") as temp:
                snapshot = Path(temp) / "calibration.json"
                snapshot.write_bytes(calibration)
                target = ["--start-home", args.start_home]
                if profile_bytes is not None:
                    profile_snapshot = Path(temp) / "task_start.json"
                    profile_snapshot.write_bytes(profile_bytes)
                    target = ["--start-profile", str(profile_snapshot)]
                check = subprocess.run([drag[0], str(ROOT / "scripts/grav_comp/check_start_pose.py"),
                                        "--midas-repo", str(midas), "--calib", str(snapshot),
                                        *target, "--json"], text=True, capture_output=True)
                if check.returncode:
                    raise RuntimeError("Start torque check failed:\n" + check.stdout + check.stderr)
                torque = json.loads(check.stdout)
            print("Start gravity commands N*m: " + " ".join(f"{x:+.3f}" for x in torque["command_nm"]), flush=True)
        else:
            print("Definition recording: no saved start target; start torque was not checked.", flush=True)
        tests = subprocess.run([drag[0], "-m", "pytest", "gravity_comp/tests", "-q", "-x"],
                               cwd=midas, text=True, capture_output=True)
        if tests.returncode:
            raise RuntimeError("Gravity-comp preflight tests failed:\n" + tests.stdout + tests.stderr)
        print(tests.stdout.strip().splitlines()[-1], flush=True)
        if args.preflight_only:
            print("Dependency preflight passed. No controller or collector started."
                  + (" Task-start torque was not checked." if args.define_start else ""))
            return 0
        print("Confirm the hand, adapter, controller, camera and cable dressing match the payload calibration.")
        if input("Type mounted when checked (anything else cancels): ").strip() != "mounted":
            return 1
        session_dir.mkdir(parents=True, exist_ok=False)
        (session_dir / "calibration.json").write_bytes(calibration)
        (session_dir / "calibration_initial.json").write_bytes(calibration)
        if profile_bytes is not None:
            (session_dir / "task_start.json").write_bytes(profile_bytes)
        torque["calibration_path"] = str(session_dir / "calibration_initial.json")
        if profile_bytes is not None:
            torque["start_profile_path"] = str(session_dir / "task_start.json")
        name = "kinesthetic-" + session_id
        pane = tmux("new-session", "-d", "-P", "-F", "#{pane_id}", "-s", name,
                    "-x", "180", "-y", "50", "-c", midas, "bash --noprofile --norc")
        tmux("set-option", "-t", name, "remain-on-exit", "on")
        other = tmux("split-window", "-h", "-P", "-F", "#{pane_id}", "-t", pane, "-c", ROOT,
                     "bash --noprofile --norc")
        manifest = dict(session_id=session_id, drag_pane=pane, collector_pane=other,
                        drag_command=drag, collector_command=collector,
                        calibration_sha256=hashlib.sha256(calibration).hexdigest(), start_check=torque,
                        episode_role="start_definition" if args.define_start else "demonstration",
                        start_profile_path=str(session_dir / "task_start.json") if profile is not None else None,
                        start_profile_sha256=profile_hash, start_profile=profile)
        (session_dir / "session.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for target, filename in ((pane, "drag.log"), (other, "collector.log")):
            tmux("pipe-pane", "-o", "-t", target, "cat >> " + shlex.quote(str(session_dir / filename)))
        supervisor = [str(ROOT / ".venv/bin/python"), "-u", str(Path(__file__).resolve()), "collect", str(session_dir)]
        for target, command in ((other, supervisor), (pane, drag)):
            tmux("send-keys", "-l", "-t", target, "exec " + shlex.join(command))
            tmux("send-keys", "-t", target, "Enter")
        tmux("select-pane", "-t", pane)
        print(f"Session: {session_dir}\nAttach: tmux attach -t {name}")
        print("Stop collector first; then q in drag, support the arm, Enter to disable. Detaching leaves drag running.")
    if not args.detach:
        try:
            subprocess.run(["tmux", "switch-client" if os.environ.get("TMUX") else "attach-session", "-t", name], check=True)
        except KeyboardInterrupt:
            print(f"Session left running: tmux attach -t {name}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
