#!/usr/bin/env python3
"""Quest -> Piper arm teleoperation together with TriMI leader -> follower hand teleoperation.

    python3 scripts/teleop/quest_trimi_teleop.py --calibration CAL.json --quest-port 5005 \\
        -- --leader-port $L --follower-port $F --force-feedback --grasp-only --log hand_$(date +%m%d%H%M)

The operator wears the leader TriMI with the Quest controller strapped to it;
the follower TriMI is mounted on the Piper flange. Options before `--` go to
scripts/teleop/quest_teleop.py (the arm), options after it to
scripts/teleop/trimi_teleop.py (the hand, force feedback included), unchanged:
see their --help. The hand's tuning YAML is read as usual.

Both scripts run unchanged, each in its own process, so the hand's 100 Hz loop
keeps its timing (the arm's CAN, logger and Quest threads would otherwise share
its interpreter lock):

1. The hand starts first. The arm starts once the follower is following, so a
   hand that fails to start (port, calibration) stops before the arm moves.
2. The arm then runs as quest_teleop.py does: Enter to move home, pedal clutch,
   h+Enter = home, q+Enter = home and quit.
3. When either stops, the other is stopped as by Ctrl-C: the hand's torque goes
   off and its log is saved; the arm holds its last target. Ctrl-C stops both.

Hand output is prefixed with [hand], this launcher's with [teleop].
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Sequence, TextIO

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.teleop import quest_teleop, trimi_teleop  # noqa: E402

TELEOP_DIR = Path(__file__).resolve().parent
ARM_SCRIPT = TELEOP_DIR / "quest_teleop.py"
HAND_SCRIPT = TELEOP_DIR / "trimi_teleop.py"
HAND_PREFIX = "[hand] "
POLL_S = 0.05
INTERRUPTED = 130  # exit code after Ctrl-C, as quest_teleop.py


def split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """(arm options, hand options): before and after the first `--`."""

    argv = list(argv)
    if "--" not in argv:
        return argv, []
    split = argv.index("--")
    return argv[:split], argv[split + 1:]


def check_args(arm_argv: Sequence[str], hand_argv: Sequence[str]) -> None:
    """Parse both sides as their scripts will, so a bad option stops here, before anything starts."""

    sides = (("arm", "before --", quest_teleop.parse_args, arm_argv),
             ("hand", "after --", trimi_teleop.parse_args, hand_argv))
    parsed = {}
    for side, where, parse, argv in sides:
        try:
            parsed[side] = parse(list(argv))
        except SystemExit as exc:
            if exc.code:
                print(f"[teleop] in the {side} options ({where}). Arm options go before --, hand options "
                      "after it.", file=sys.stderr)
            raise
    if parsed["arm"].set_home_here:
        raise SystemExit("[teleop] --set-home-here only saves the arm's home: run quest_teleop.py on its own.")
    if parsed["hand"].save_tuning is not None:
        raise SystemExit("[teleop] --save-tuning only writes a file: run trimi_teleop.py on its own.")


def _pump(stream: TextIO, out: TextIO, prefix: str, marker: str, seen: threading.Event) -> None:
    """Copy a child's output to `out` with a prefix until it closes; set `seen` at the line holding `marker`.

    Reads to the end whatever happens to `out`: a child whose pipe fills up would block on its next print.
    """

    for line in stream:
        if marker in line:
            seen.set()
        try:
            out.write(prefix + line)
            out.flush()
        except (OSError, ValueError):
            pass


class Child:
    """A started script; stop() sends it one SIGINT, so it shuts down as on Ctrl-C."""

    def __init__(self, name: str, process: subprocess.Popen):
        self.name, self.process = name, process
        self.stop_sent = False

    def running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.running() and not self.stop_sent:
            self.stop_sent = True
            self.process.send_signal(signal.SIGINT)


def supervise(arm_command: Sequence[str], hand_command: Sequence[str], *, out: TextIO | None = None,
              ready_marker: str = trimi_teleop.FOLLOWING_MESSAGE,
              say: Callable[[str], None] | None = None) -> int:
    """Start the hand, then (once it prints `ready_marker`) the arm; stop both when either stops.

    Returns the exit code of the one that stopped first if it failed, 130 after Ctrl-C, else 0.
    Ctrl-C at the terminal reaches every process of the group, so it is not passed on (a second SIGINT
    would cut a child's shutdown short); SIGTERM and SIGHUP are passed on as one SIGINT each.
    """

    out = sys.stdout if out is None else out
    say = say or (lambda text: print(f"[teleop] {text}", flush=True))
    children: list[Child] = []
    interrupted = threading.Event()
    terminated = threading.Event()

    def on_interrupt(signum, frame):
        interrupted.set()

    def on_terminate(signum, frame):
        terminated.set()
        for child in children:
            child.stop()

    def stopping() -> bool:
        return interrupted.is_set() or terminated.is_set()

    handlers = {signal.SIGINT: on_interrupt, signal.SIGTERM: on_terminate}
    if hasattr(signal, "SIGHUP"):
        handlers[signal.SIGHUP] = on_terminate
    previous = {sig: signal.signal(sig, handler) for sig, handler in handlers.items()}
    pump = None
    try:
        say("Starting the hand; the arm starts once the follower is following.")
        hand = Child("hand", subprocess.Popen(
            list(hand_command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, errors="replace", bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"}))
        children.append(hand)
        ready = threading.Event()
        pump = threading.Thread(target=_pump, args=(hand.process.stdout, out, HAND_PREFIX, ready_marker, ready),
                                name="hand-output", daemon=True)
        pump.start()
        while not ready.is_set() and hand.running() and not stopping():
            time.sleep(POLL_S)
        if not ready.is_set() or not hand.running() or stopping():
            code = hand.process.wait()
            if stopping():
                return INTERRUPTED
            say(f"The hand stopped (exit {code}) before the arm was started.")
            return code or 1

        # The arm keeps the terminal: its Enter prompt and h/q commands read stdin.
        arm = Child("arm", subprocess.Popen(list(arm_command)))
        children.append(arm)
        while hand.running() and arm.running():
            time.sleep(POLL_S)
        first, other = (hand, arm) if not hand.running() else (arm, hand)
        code = first.process.returncode
        if other.running() and not stopping():
            outcome = "the arm holds its last target" if other is arm else "the hand's torque goes off"
            say(f"The {first.name} stopped (exit {code}); stopping the {other.name} ({outcome}).")
            other.stop()
        other.process.wait()
        if stopping():
            return INTERRUPTED
        if code:
            return code
        return 0 if other.stop_sent else other.process.returncode
    finally:
        for child in children:  # an error above: never leave a child running unwatched
            if child.running():
                child.stop()
                child.process.wait()
        if pump is not None:
            pump.join(timeout=5.0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: Sequence[str] | None = None) -> int:
    arm_argv, hand_argv = split_argv(sys.argv[1:] if argv is None else argv)
    if {"-h", "--help"} & set(arm_argv):
        print(__doc__)
        print("Arm options: python3 scripts/teleop/quest_teleop.py --help\n"
              "Hand options: python3 scripts/teleop/trimi_teleop.py --help")
        return 0
    check_args(arm_argv, hand_argv)
    return supervise([sys.executable, str(ARM_SCRIPT), *arm_argv],
                     [sys.executable, str(HAND_SCRIPT), *hand_argv])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
