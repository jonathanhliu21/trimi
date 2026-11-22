#!/usr/bin/env python3
"""Check TriMi hand connectivity and per-motor status.

Read-only by default: it pings every motor, prints a status table (model,
firmware, mode, torque, hardware errors, voltage, temperature, position,
current) and benchmarks the sync-read rate. Nothing moves and nothing is
written to the motors unless an opt-in flag is given:

  --motion-test            moves each motor +/- a small angle (hand must be EMPTY)
  --set-return-delay-zero  writes Return Delay Time = 0 to motor EEPROM

Example:
  python smoke_test.py
  python smoke_test.py --motion-test
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rss_common as rc  # noqa: E402


ADDR_FIRMWARE = 6
ADDR_RETURN_DELAY = 9
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_HARDWARE_ERROR = 70
ADDR_INPUT_VOLTAGE = 144
ADDR_TEMPERATURE = 146

DEFAULT_BENCH_SAMPLES = 200
MAX_TEMPERATURE_C = 60
VOLTAGE_RANGE_V = (4.5, 5.8)
MODE_NAMES = {0: "current", 1: "velocity", 3: "position", 4: "ext-pos", 5: "cur-pos", 16: "pwm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=rc.DEFAULT_CONFIG, help="Hand YAML (default: %(default)s)")
    parser.add_argument("--port", default=rc.DEFAULT_PORT, help="Serial port (default: %(default)s)")
    parser.add_argument("--bench-samples", type=int, default=DEFAULT_BENCH_SAMPLES, help="Sync reads to time (default: %(default)s)")
    parser.add_argument("--motion-test", action="store_true", help="Move each motor a small angle and back (hand must be empty)")
    parser.add_argument("--motion-amplitude-rad", type=float, default=0.15, help="Motion test amplitude (default: %(default)s rad)")
    parser.add_argument("--motion-current-ma", type=int, default=300, help="Current cap during the motion test (default: %(default)s mA)")
    parser.add_argument("--motion-tolerance-rad", type=float, default=0.05, help="Pass threshold on settled error (default: %(default)s rad)")
    parser.add_argument("--set-return-delay-zero", action="store_true", help="Write Return Delay Time = 0 (EEPROM) on motors where it is nonzero")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts for opt-in actions")
    return parser.parse_args()


class RegisterReader:
    """Single-register reads through the hand's own SDK handles."""

    def __init__(self, hand) -> None:
        client = hand._client()
        self.port = client.port_handler
        self.packet = client.packet_handler

    def read(self, motor_id: int, address: int, size: int) -> int | None:
        fn = {1: self.packet.read1ByteTxRx, 2: self.packet.read2ByteTxRx, 4: self.packet.read4ByteTxRx}[size]
        value, comm_result, _error = fn(self.port, int(motor_id), int(address))
        return int(value) if comm_result == 0 else None

    def write1(self, motor_id: int, address: int, value: int) -> bool:
        comm_result, error = self.packet.write1ByteTxRx(self.port, int(motor_id), int(address), int(value))
        return comm_result == 0 and error == 0


def fmt(value, spec: str = "", missing: str = "ERR") -> str:
    return missing if value is None else format(value, spec)


def status_table(hand, models: dict[int, int]) -> tuple[list[str], dict[int, int]]:
    reader = RegisterReader(hand)
    positions, _velocities, currents = hand.read_pos_vel_cur()
    lower = hand.config.joint_lower_limits_array
    upper = hand.config.joint_upper_limits_array
    problems: list[str] = []
    return_delays: dict[int, int] = {}

    header = (
        f"{'ID':>2}  {'joint':<17} {'model':<13} {'fw':>3} {'RDT':>4} {'mode':<8} "
        f"{'torque':<6} {'hw error':<16} {'volt':>5} {'temp':>4} {'pos rad':>8} {'limits':<15} {'cur mA':>7}"
    )
    print(header)
    print("-" * len(header))
    for index, motor_id in enumerate(hand.motor_ids):
        model = models.get(motor_id)
        firmware = reader.read(motor_id, ADDR_FIRMWARE, 1)
        return_delay = reader.read(motor_id, ADDR_RETURN_DELAY, 1)
        mode = reader.read(motor_id, ADDR_OPERATING_MODE, 1)
        torque = reader.read(motor_id, ADDR_TORQUE_ENABLE, 1)
        hw_error = reader.read(motor_id, ADDR_HARDWARE_ERROR, 1)
        voltage_raw = reader.read(motor_id, ADDR_INPUT_VOLTAGE, 2)
        temperature = reader.read(motor_id, ADDR_TEMPERATURE, 1)
        voltage = None if voltage_raw is None else voltage_raw / 10.0
        if return_delay is not None:
            return_delays[motor_id] = return_delay

        q = float(positions[index])
        in_limits = lower[index] <= q <= upper[index]
        limit_text = f"[{lower[index]:.2f},{upper[index]:.2f}]" if upper[index] - lower[index] < 50 else "open"
        print(
            f"{motor_id:>2}  {rc.motor_label(motor_id):<17} {rc.MODEL_NAMES.get(model, str(model)):<13} "
            f"{fmt(firmware):>3} {fmt(return_delay):>4} {MODE_NAMES.get(mode, fmt(mode)):<8} "
            f"{('on' if torque else 'off') if torque is not None else 'ERR':<6} "
            f"{rc.decode_hardware_error(hw_error) if hw_error is not None else 'ERR':<16} "
            f"{fmt(voltage, '.1f'):>5} {fmt(temperature):>4} {q:>8.3f} "
            f"{limit_text + ('' if in_limits else ' OUT'):<15} {float(currents[index]):>7.0f}"
        )

        tag = f"ID {motor_id} ({rc.motor_label(motor_id)})"
        if None in (firmware, mode, torque, hw_error, voltage_raw, temperature):
            problems.append(f"{tag}: a status register read failed")
        if hw_error:
            problems.append(f"{tag}: hardware error {rc.decode_hardware_error(hw_error)}")
        if temperature is not None and temperature > MAX_TEMPERATURE_C:
            problems.append(f"{tag}: temperature {temperature} C > {MAX_TEMPERATURE_C} C")
        if voltage is not None and not VOLTAGE_RANGE_V[0] <= voltage <= VOLTAGE_RANGE_V[1]:
            problems.append(f"{tag}: voltage {voltage:.1f} V outside {VOLTAGE_RANGE_V}")
        if not in_limits:
            print(f"  note: {tag} is outside its configured joint limits (commands would be clipped)")
    return problems, return_delays


def benchmark(hand, samples: int) -> dict[str, float]:
    durations = []
    with rc.ReadErrorCounter() as errors:
        for _ in range(samples):
            start = time.perf_counter()
            hand.read_pos_vel_cur()
            durations.append((time.perf_counter() - start) * 1e3)
    durations.sort()
    median = statistics.median(durations)
    return {
        "median_ms": median,
        "p95_ms": durations[int(0.95 * (len(durations) - 1))],
        "max_ms": durations[-1],
        "rate_hz": 1000.0 / median,
        "read_errors": errors.count,
    }


def print_benchmark(result: dict[str, float], samples: int) -> None:
    print(
        f"read_pos_vel_cur x{samples}: median {result['median_ms']:.2f} ms, "
        f"p95 {result['p95_ms']:.2f} ms, max {result['max_ms']:.2f} ms "
        f"-> ~{result['rate_hz']:.0f} Hz, read errors {result['read_errors']}"
    )


def set_return_delay_zero(hand, return_delays: dict[int, int], assume_yes: bool) -> None:
    targets = [motor_id for motor_id, value in return_delays.items() if value != 0]
    if not targets:
        print("Return Delay Time is already 0 on every motor.")
        return
    print(f"Motors with nonzero Return Delay Time: {targets}")
    if not assume_yes and not confirm_eeprom():
        print("Skipped Return Delay Time change.")
        return
    hand.disable_torque(verify=True)  # EEPROM writes require torque off
    reader = RegisterReader(hand)
    for motor_id in targets:
        ok = reader.write1(motor_id, ADDR_RETURN_DELAY, 0)
        after = reader.read(motor_id, ADDR_RETURN_DELAY, 1)
        print(f"  ID {motor_id}: write {'ok' if ok else 'FAILED'}, now {after}")


def confirm_eeprom() -> bool:
    return rc.confirm("Write Return Delay Time = 0 to these motors' EEPROM (persists across power cycles)?")


def cosine_ramp(hand, start: np.ndarray, goal: np.ndarray, duration_s: float, rate_hz: float = 100.0) -> float:
    """Move from start to goal; return peak |current| (mA) of the moving motor set."""

    clock = rc.FixedRateClock(rate_hz)
    clock.start()
    peak = 0.0
    while True:
        elapsed = clock.wait()
        alpha = min(elapsed / duration_s, 1.0)
        blend = 0.5 - 0.5 * math.cos(math.pi * alpha)
        hand.set_positions(start + blend * (goal - start), clip=False)
        _q, _v, current = hand.read_pos_vel_cur()
        peak = max(peak, float(np.max(np.abs(current))))
        if alpha >= 1.0:
            return peak


def motion_test(hand, args: argparse.Namespace) -> list[str]:
    print(
        "\nMOTION TEST: every motor moves "
        f"{args.motion_amplitude_rad:.2f} rad and back, mode 5, {args.motion_current_ma} mA cap."
    )
    print("The hand must be EMPTY and not worn.")
    if not args.yes and not rc.confirm("Start the motion test?"):
        print("Skipped motion test.")
        return []

    hand.config = replace(hand.config, operating_mode=5, goal_current_limit=int(args.motion_current_ma))
    hand.configure(enable_torque=False)
    hand.set_motion_profile(None, None)
    hold = hand.read_pos()
    hand.set_positions(hold, clip=False)
    hand.enable_torque()
    lower = hand.config.joint_lower_limits_array
    upper = hand.config.joint_upper_limits_array
    failures: list[str] = []
    try:
        for index, motor_id in enumerate(hand.motor_ids):
            start = hand.read_pos()
            start[np.arange(len(start)) != index] = hold[np.arange(len(hold)) != index]
            goal = start.copy()
            for direction in (1.0, -1.0):
                candidate = float(np.clip(start[index] + direction * args.motion_amplitude_rad, lower[index], upper[index]))
                if abs(candidate - start[index]) >= 0.5 * args.motion_amplitude_rad:
                    goal[index] = candidate
                    break
            else:
                failures.append(f"ID {motor_id}: no room within joint limits to move")
                print(f"  ID {motor_id:>2} {rc.motor_label(motor_id):<17} SKIP (at joint limits)")
                continue
            peak = cosine_ramp(hand, start, goal, 1.0)
            time.sleep(0.3)
            reached = hand.read_pos()[index]
            error = abs(reached - goal[index])
            peak = max(peak, cosine_ramp(hand, goal, start, 1.0))
            ok = error <= args.motion_tolerance_rad
            print(
                f"  ID {motor_id:>2} {rc.motor_label(motor_id):<17} "
                f"{start[index]:+.3f} -> {goal[index]:+.3f} rad, settled error {error:.3f} rad, "
                f"peak |I| {peak:.0f} mA  {'PASS' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"ID {motor_id}: settled error {error:.3f} rad > {args.motion_tolerance_rad} rad")
    finally:
        hand.disable_torque(verify=True)
        print("Torque OFF.")
    return failures


def main() -> int:
    args = parse_args()
    print(f"Port: {args.port} -> {Path(args.port).resolve()}")
    print(f"Config: {args.config}")
    print(f"USB latency timer before connect: {rc.latency_timer_ms(args.port)} ms")

    config = rc.load_config(args.config, args.port)
    from midas_hand_api import MidasHand

    hand = MidasHand(config, autoconnect=True)
    problems: list[str] = []
    try:
        latency = rc.latency_timer_ms(args.port)
        print(f"USB latency timer after connect:  {latency} ms")
        if latency is not None and latency > 1:
            problems.append(f"USB latency timer is {latency} ms (expected 1 ms)")

        models = hand.ping()
        missing = sorted(set(hand.motor_ids) - set(models))
        if missing:
            problems.append(f"missing motor IDs: {missing}")
            print(f"Missing motor IDs: {missing}; re-running checks on responders only.")
            hand.close()
            hand = MidasHand(config.subset(sorted(models)), autoconnect=True)
        wrong = {m: n for m, n in models.items() if n not in rc.COMPATIBLE_MODEL_NUMBERS}
        if wrong:
            problems.append(f"unexpected motor models: {wrong}")
        print(f"{len(models)}/{len(config.motor_ids)} motors responded.\n")

        table_problems, return_delays = status_table(hand, models)
        problems.extend(table_problems)
        if any(return_delays.values()):
            print(
                "\nnote: Return Delay Time is nonzero on "
                f"{[m for m, v in return_delays.items() if v]} "
                "(each adds 2 us per unit to every read; --set-return-delay-zero removes it)"
            )

        print()
        result = benchmark(hand, args.bench_samples)
        print_benchmark(result, args.bench_samples)
        if result["read_errors"]:
            problems.append(f"{result['read_errors']} sync-read errors during benchmark")

        if args.set_return_delay_zero:
            print()
            set_return_delay_zero(hand, return_delays, args.yes)
            print_benchmark(benchmark(hand, args.bench_samples), args.bench_samples)

        if args.motion_test:
            problems.extend(motion_test(hand, args))
    finally:
        hand.shutdown()

    print()
    if problems:
        print("SMOKE TEST: PROBLEMS FOUND")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("SMOKE TEST: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
