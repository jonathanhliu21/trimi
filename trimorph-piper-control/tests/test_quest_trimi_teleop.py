import io
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from scripts.teleop.quest_trimi_teleop import check_args, split_argv, supervise
from scripts.teleop.trimi_teleop import FOLLOWING_MESSAGE


def script(body):
    """A fake teleop script: `python -c body`, with an `until_stopped()` helper (Ctrl-C ends it)."""

    prelude = textwrap.dedent("""
        import sys, time
        def until_stopped(code):
            try:
                while True:
                    time.sleep(0.01)
            except KeyboardInterrupt:
                print("stopped", flush=True)
                sys.exit(code)
    """)
    return [sys.executable, "-c", prelude + textwrap.dedent(body)]


class SplitTests(unittest.TestCase):
    def test_options_before_the_separator_are_the_arm_s(self):
        self.assertEqual(split_argv(["--calibration", "c.json", "--", "--force-feedback", "--", "x"]),
                         (["--calibration", "c.json"], ["--force-feedback", "--", "x"]))

    def test_without_separator_the_hand_gets_nothing(self):
        self.assertEqual(split_argv(["--calibration", "c.json"]), (["--calibration", "c.json"], []))

    def test_one_shot_modes_and_misplaced_options_are_refused(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                check_args(["--calibration", "c.json", "--set-home-here"], ["--no-tuning"])
            with self.assertRaises(SystemExit):
                check_args(["--calibration", "c.json"], ["--no-tuning", "--save-tuning", "t.yaml"])
            with self.assertRaises(SystemExit):  # hand options before the separator
                check_args(["--calibration", "c.json", "--force-feedback"], ["--no-tuning"])
        check_args(["--calibration", "c.json"], ["--no-tuning", "--force-feedback", "--grasp-only"])


class SuperviseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.arm_started = self.tmp / "arm_started"
        self.out = io.StringIO()
        self.said = []

    def run_pair(self, arm_body, hand_body):
        arm = script(f"open({str(self.arm_started)!r}, 'w').close()\n" + textwrap.dedent(arm_body))
        start = time.monotonic()
        code = supervise(arm, script(hand_body), out=self.out, say=self.said.append)
        self.assertLess(time.monotonic() - start, 10.0)
        return code

    def test_arm_waits_for_the_hand_and_its_quit_stops_the_hand(self):
        code = self.run_pair(
            "import time; time.sleep(0.2); sys.exit(0)",
            f"print('setting up', flush=True); time.sleep(0.3)\n"
            f"print({FOLLOWING_MESSAGE + ' over 2 s'!r}, flush=True); until_stopped(0)")
        self.assertEqual(code, 0)
        self.assertTrue(self.arm_started.exists())
        lines = self.out.getvalue().splitlines()
        self.assertEqual(lines[0], "[hand] setting up")
        self.assertEqual(lines[-1], "[hand] stopped")  # the hand got its Ctrl-C and shut down
        self.assertTrue(any("stopping the hand" in s for s in self.said))

    def test_hand_failing_at_start_never_starts_the_arm(self):
        code = self.run_pair("until_stopped(130)", "print('Error: no such port', flush=True); sys.exit(1)")
        self.assertEqual(code, 1)
        self.assertFalse(self.arm_started.exists())
        self.assertIn("[hand] Error: no such port", self.out.getvalue())

    def test_hand_failing_while_teleoperating_stops_the_arm(self):
        code = self.run_pair(
            "until_stopped(130)",
            f"print({FOLLOWING_MESSAGE!r}, flush=True); time.sleep(0.5)\n"
            "print('Error: Leader joint 3 jumped', flush=True); sys.exit(3)")
        self.assertEqual(code, 3)
        self.assertTrue(self.arm_started.exists())
        self.assertTrue(any("stopping the arm" in s for s in self.said))

    def test_chatty_hand_never_blocks_on_its_output(self):
        code = self.run_pair(
            "import time; time.sleep(2.0); sys.exit(0)",
            f"print({FOLLOWING_MESSAGE!r}, flush=True)\n"
            "for i in range(20000): print('leader deg ' + 'x' * 60)\n"  # far more than a pipe holds
            "until_stopped(0)")
        self.assertEqual(code, 0)
        self.assertGreaterEqual(self.out.getvalue().count("[hand] leader deg"), 20000)


if __name__ == "__main__":
    unittest.main()
