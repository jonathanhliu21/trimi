"""Use a task's saved start with the existing policy executor."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_collection.collection.config import KINESTHETIC_HAND_CONFIG
from inference.executor.executor import parse_args
from scripts.kinesthetic.launch import running_drag


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not any(value in ("--start-profile", "--start-home", "--start-joints-deg", "--start-id")
               or value.startswith(("--start-profile=", "--start-home=", "--start-joints-deg=", "--start-id="))
               for value in argv):
        raise ValueError("Choose this task's --start-profile PATH (or an explicit legacy start); no fixed task start is selected")
    if not any(value == "--hand-config" or value.startswith("--hand-config=") for value in argv):
        argv = ["--hand-config", str(ROOT / KINESTHETIC_HAND_CONFIG), *argv]
    args = parse_args(argv)
    if not args.fake and running_drag(args.piper_channel):
        raise RuntimeError("Stop drag mode with q, support the arm, then Enter before starting the executor")
    os.execv(sys.executable, [sys.executable, "-u", str(ROOT / "inference/executor/executor.py"), *argv])


if __name__ == "__main__":
    main()
