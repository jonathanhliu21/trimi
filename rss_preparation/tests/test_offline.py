"""Offline tests for the RSS record/replay/plot scripts (no hardware)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

RSS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RSS_DIR))

import plot_trials  # noqa: E402
import replay_trajectory as rt  # noqa: E402
import rss_common as rc  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return rc.load_config(rc.DEFAULT_CONFIG, None)


class FakeHand:
    """First-order position lag; current proportional to tracking error."""

    def __init__(self, config, *, lag=0.5, gain_ma_per_rad=1000.0, forced_current_ma=None):
        self.config = config
        self.motor_ids = list(config.motor_ids)
        self.q = np.zeros(len(self.motor_ids))
        self.command = None
        self.lag = lag
        self.gain = gain_ma_per_rad
        self.forced_current_ma = forced_current_ma
        self.commands: list[np.ndarray] = []
        self.disable_calls = 0

    def set_positions(self, positions, clip=True):
        q = np.asarray(positions, dtype=float)
        if clip:
            q = self.clip_positions(q)
        self.command = q
        self.commands.append(q.copy())
        self.q = self.q + self.lag * (q - self.q)

    def read_pos(self):
        return self.q.copy()

    def read_pos_vel_cur(self):
        if self.forced_current_ma is not None:
            current = np.full(len(self.q), float(self.forced_current_ma))
        elif self.command is None:
            current = np.zeros(len(self.q))
        else:
            current = self.gain * (self.command - self.q)
        return self.q.copy(), np.zeros(len(self.q)), current

    def clip_positions(self, positions):
        return np.clip(
            np.asarray(positions, dtype=float),
            self.config.joint_lower_limits_array,
            self.config.joint_upper_limits_array,
        )

    def disable_torque(self, **_kwargs):
        self.disable_calls += 1


def synthetic_trajectory(duration_s=0.3, samples=31, motors=13):
    times = np.linspace(0.0, duration_s, samples)
    phases = np.arange(motors)[None, :] * 0.3
    positions = 0.5 * np.sin(2 * math.pi * times[:, None] / duration_s + phases)
    return times, positions


# ----------------------------------------------------------------- helpers


def test_interpolate_matches_samples_midpoints_and_clamps():
    times = np.array([0.0, 1.0, 2.0])
    positions = np.array([[0.0, 0.0], [1.0, 10.0], [3.0, 30.0]])
    assert np.allclose(rt.interpolate(times, positions, 0.5), [0.5, 5.0])
    assert np.allclose(rt.interpolate(times, positions, 1.0), [1.0, 10.0])
    assert np.allclose(rt.interpolate(times, positions, 1.5), [2.0, 20.0])
    assert np.allclose(rt.interpolate(times, positions, -1.0), [0.0, 0.0])
    assert np.allclose(rt.interpolate(times, positions, 9.0), [3.0, 30.0])


def test_unwrap_turns_removes_whole_revolutions_only():
    start = np.array([0.0, 1.0, -2.0])
    measured = np.array([2 * math.pi + 0.1, 1.0 - 2 * math.pi, -2.0 + 0.3])
    turns, residual = rt.unwrap_turns(start, measured)
    assert turns.tolist() == [1, -1, 0]
    assert np.allclose(residual, [0.1, 0.0, 0.3])


def test_abort_monitor_needs_consecutive_samples():
    monitor = rt.CurrentAbortMonitor([0, 1], threshold_ma=100.0, samples=3)
    assert monitor.update([150, 0]) is None
    assert monitor.update([150, 0]) is None
    assert monitor.update([50, 0]) is None  # resets the run
    assert monitor.update([-150, 0]) is None  # magnitude counts
    assert monitor.update([150, 0]) is None
    tripped = monitor.update([150, 0])
    assert tripped["motor_id"] == 0 and tripped["current_ma"] == 150


def test_smoothing_is_zero_phase_bounded_and_attenuates_high_frequency():
    times = np.arange(0.0, 4.0, 0.01)
    slow = 0.8 * np.sin(2 * math.pi * 0.5 * times)
    fast = 0.2 * np.sin(2 * math.pi * 20.0 * times)
    positions = np.column_stack([slow + fast, slow])
    out_times, out = rc.smooth_trajectory(times, positions, cutoff_hz=5.0)

    assert np.allclose(out_times, times)
    interior = slice(100, -100)
    # 20 Hz component strongly attenuated; 0.5 Hz motion kept with no lag.
    assert np.abs(out[interior, 0] - slow[interior]).max() < 0.01
    assert np.abs(out[interior, 1] - slow[interior]).max() < 0.005
    lag = np.argmax(np.correlate(out[:, 1], slow, "full")) - (len(slow) - 1)
    assert lag == 0
    # Gaussian weights are non-negative: never outside the recorded range.
    assert out[:, 0].max() <= positions[:, 0].max() + 1e-12
    assert out[:, 0].min() >= positions[:, 0].min() - 1e-12
    # Cutoff semantics: a 5 Hz sine keeps ~1/sqrt(2) of its amplitude.
    five = np.sin(2 * math.pi * 5.0 * times)[:, None]
    _, five_out = rc.smooth_trajectory(times, five, cutoff_hz=5.0)
    assert np.abs(five_out[interior, 0]).max() == pytest.approx(1 / math.sqrt(2), abs=0.02)


def test_smoothing_disabled_returns_input():
    times, positions = synthetic_trajectory()
    out_times, out = rc.smooth_trajectory(times, positions, cutoff_hz=0.0)
    assert out_times is times and out is positions


def test_load_config_gain_overrides(config):
    tuned = rc.load_config(rc.DEFAULT_CONFIG, None, p_gain=1500, i_gain=0, d_gain=600)
    assert (tuned.position_p_gain, tuned.position_i_gain, tuned.position_d_gain) == (1500, 0, 600)
    untouched = rc.load_config(rc.DEFAULT_CONFIG, None)
    assert untouched.position_i_gain == config.position_i_gain
    with pytest.raises(ValueError):
        rc.load_config(rc.DEFAULT_CONFIG, None, p_gain=-1)


# ----------------------------------------------------------------- replay


def test_run_replay_logs_every_tick_in_telemetry_schema(config, tmp_path):
    times, positions = synthetic_trajectory()
    hand = FakeHand(config)
    monitor = rt.CurrentAbortMonitor(hand.motor_ids, 1e9, 3)

    result = rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=1.0, monitor=monitor)

    assert result.aborted is None and not result.interrupted
    assert len(result.rows) >= 0.3 * 200 * 0.8
    assert list(result.rows[0]) == rc.telemetry_fieldnames(hand.motor_ids)
    elapsed = np.array([float(r["elapsed_s"]) for r in result.rows])
    assert elapsed[0] < 0.02 and elapsed[-1] == pytest.approx(0.3)
    assert np.all(np.diff(elapsed) > 0)
    # Logged command equals the trajectory at the logged time, unclipped.
    for row, command in zip(result.rows, hand.commands):
        expected = rt.interpolate(times, positions, float(row["elapsed_s"]))
        logged = [float(row[f"motor_{m}_target_position_rad"]) for m in hand.motor_ids]
        assert np.allclose(logged, command, atol=1e-6)  # log == what was sent
        assert np.allclose(logged, expected, atol=1e-4)  # elapsed_s is rounded to 1 us

    path = tmp_path / "idle.csv"
    rc.write_csv(path, rc.telemetry_fieldnames(hand.motor_ids), result.rows)
    telemetry = rc.load_telemetry(path)
    assert telemetry.motor_ids == tuple(hand.motor_ids)
    assert telemetry.current.shape == (len(result.rows), 13)


def test_run_replay_playback_speed_scales_time(config):
    times, positions = synthetic_trajectory(duration_s=0.4, samples=41)
    hand = FakeHand(config)
    monitor = rt.CurrentAbortMonitor(hand.motor_ids, 1e9, 3)
    result = rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=2.0, monitor=monitor)
    stamps = [int(r["monotonic_ns"]) for r in result.rows]
    wall_s = (stamps[-1] - stamps[0]) / 1e9
    assert float(result.rows[-1]["elapsed_s"]) == pytest.approx(0.4)
    assert wall_s == pytest.approx(0.2, abs=0.05)


def test_run_replay_aborts_and_disables_torque(config):
    times, positions = synthetic_trajectory()
    hand = FakeHand(config, forced_current_ma=2000.0)
    monitor = rt.CurrentAbortMonitor(hand.motor_ids, 1000.0, 3)

    result = rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=1.0, monitor=monitor)

    assert result.aborted is not None and result.aborted["current_ma"] == 2000.0
    assert len(result.rows) == 3
    assert hand.disable_calls == 1


def test_run_replay_clip_option(config):
    times = np.array([0.0, 0.05])
    positions = np.full((2, 13), 0.5)
    from dataclasses import replace

    limited = replace(config, joint_lower_limits=[-1.0] * 13, joint_upper_limits=[0.0] * 13)
    hand = FakeHand(limited)
    monitor = rt.CurrentAbortMonitor(hand.motor_ids, 1e9, 3)
    rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=1.0, monitor=monitor, clip=True)
    pip = hand.motor_ids.index(4)
    assert hand.commands[-1][pip] == pytest.approx(0.0)
    rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=1.0, monitor=monitor, clip=False)
    assert hand.commands[-1][pip] == pytest.approx(0.5)


# ----------------------------------------------------------------- schemas


def test_trajectory_schema_matches_collection_and_roundtrips(tmp_path):
    motor_ids = list(range(13))
    assert rc.trajectory_fieldnames(motor_ids)[:4] == ["sample_idx", "monotonic_ns", "wall_time_s", "elapsed_s"]
    times, positions = synthetic_trajectory()
    rows = []
    for i, (t, q) in enumerate(zip(times, positions)):
        row = {"sample_idx": i, "monotonic_ns": 1_000 + i, "wall_time_s": 0.0, "elapsed_s": t + 5.0}
        row.update({f"pos_{m}": q[j] for j, m in enumerate(motor_ids)})
        rows.append(row)
    path = tmp_path / rc.TRAJECTORY_FILENAME
    rc.write_csv(path, rc.trajectory_fieldnames(motor_ids), rows)
    loaded_ids, loaded_times, loaded_positions = rc.load_trajectory(path)
    assert loaded_ids == tuple(motor_ids)
    assert np.allclose(loaded_times, times)  # re-zeroed
    assert np.allclose(loaded_positions, positions)


def test_replay_csv_readable_by_existing_graph_tool(config, tmp_path):
    repo = RSS_DIR.parent / "trimorph-piper-control"
    sys.path.insert(0, str(repo))
    try:
        from scripts.trimorph_api_test.gen_graphs_from_csvs import _load_telemetry
    finally:
        sys.path.remove(str(repo))
    times, positions = synthetic_trajectory()
    hand = FakeHand(config)
    result = rt.run_replay(hand, times, positions, rate_hz=200.0, playback_speed=1.0, monitor=rt.CurrentAbortMonitor(hand.motor_ids, 1e9, 3))
    path = tmp_path / "idle.csv"
    rc.write_csv(path, rc.telemetry_fieldnames(hand.motor_ids), result.rows)
    trials, loaded_times, motors = _load_telemetry(path)
    assert set(motors) == set(range(13))
    assert len(loaded_times) == len(result.rows)


# ----------------------------------------------------------------- plotting


def write_synthetic_session(root: Path, config) -> dict[str, float]:
    """Three conditions with known constant tracking offsets and currents."""

    offsets = {"idle": 0.01, "tpu": 0.02, "human": 0.04}
    currents = {"idle": 50.0, "tpu": 150.0, "human": 300.0}
    motor_ids = list(config.motor_ids)
    times = np.linspace(0.0, 2.0, 201)
    snapshot = rc.config_snapshot(config)
    for condition, offset in offsets.items():
        rows = []
        for t in times:
            target = 0.4 * np.sin(t + np.arange(13))
            row = {"trial": 1, "elapsed_s": t, "monotonic_ns": int(t * 1e9)}
            for i, m in enumerate(motor_ids):
                row[f"motor_{m}_current_ma"] = currents[condition]
                row[f"motor_{m}_target_position_rad"] = target[i]
                row[f"motor_{m}_actual_position_rad"] = target[i] + offset
                row[f"motor_{m}_target_velocity_rad_s"] = 0.0
                row[f"motor_{m}_actual_velocity_rad_s"] = 0.0
            rows.append(row)
        paths = rc.SessionPaths(root)
        rc.write_csv(paths.replay_csv(condition), rc.telemetry_fieldnames(motor_ids), rows)
        rc.write_json(
            paths.replay_metadata(condition),
            {
                "condition": condition,
                "complete": True,
                "trajectory_sha256": "abc",
                "settings": {"operating_mode": 5, "current_limit_ma": 1700},
                "timing": {"achieved_rate_hz": 100.0},
                "config": snapshot,
            },
        )
    return offsets


def test_plot_trials_writes_figures_and_summary(config, tmp_path, monkeypatch, capsys):
    session = tmp_path / "demo"
    offsets = write_synthetic_session(session, config)
    monkeypatch.setattr(sys, "argv", ["plot_trials.py", str(session)])

    assert plot_trials.main() == 0

    figures = session / "figures"
    for name in ("current_overlay.png", "current_overlay.pdf", "position_overlay.png", "position_overlay.pdf", "summary.csv"):
        assert (figures / name).stat().st_size > 0, name
    import csv

    with (figures / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 14  # 13 joints + mean row per condition
    for row in rows:
        assert float(row["rms_error_rad"]) == pytest.approx(offsets[row["condition"]], abs=1e-6)
    assert "RMS tracking error" in capsys.readouterr().out


def test_plot_marks_partial_runs(config, tmp_path, monkeypatch, capsys):
    session = tmp_path / "demo"
    write_synthetic_session(session, config)
    meta_path = rc.SessionPaths(session).replay_metadata("human")
    meta = json.loads(meta_path.read_text())
    meta["complete"] = False
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(sys, "argv", ["plot_trials.py", str(session), "--formats", "png"])
    assert plot_trials.main() == 0
    assert "partial run" in capsys.readouterr().out
