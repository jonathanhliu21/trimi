"""Offline log diagnostics must retain evidence and handle incomplete data."""
import json

import numpy as np
import pytest

from scripts.grav_comp.compare_drag_response import main, summarize


def write_log(path, **overrides):
    t = np.arange(0.0, 8.0, 0.1)
    q = np.tile(t[:, None] / 100.0, (1, 6))
    data = dict(t=t, q=q, tau_cmd=np.ones_like(q),
                tau_fb=5.0 - 150.0 * q,
                state=np.where(t < 4, "RAMP_IN", "ACTIVE"))
    data.update(overrides)
    np.savez(path, **data)
    return path


def test_reports_ramp_and_restorative_association_without_a_physical_pass(tmp_path):
    report = summarize(write_log(tmp_path / "test.npz"))
    early = report["ramp_response"]["first_0_to_0_5_s"]
    assert early["ramp_samples"] == 5
    assert early["mean_command_nm"] == [1.0] * 6
    joint = report["active_joint_diagnostics"][0]
    assert joint["active_command_range_nm"] == 0
    assert joint["feedback_vs_angle_slope_nm_per_rad"] == pytest.approx(-150)
    assert joint["feedback_vs_angle_correlation"] == pytest.approx(-1)
    assert "pass" not in report


def test_nonfinite_feedback_is_counted_and_excluded_without_json_nan(tmp_path, capsys):
    path = write_log(tmp_path / "finite.npz")
    with np.load(path) as log:
        feedback = log["tau_fb"].copy()
    feedback[45, 0] = np.nan
    write_log(path, tau_fb=feedback)
    output = tmp_path / "report.json"
    assert main([str(path), str(path), "--output", str(output)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == json.loads(output.read_text())
    retry = stdout["retry"]
    assert retry["nonfinite_values"]["tau_fb"] == 1
    assert retry["active_joint_diagnostics"][0]["fit_samples"] == 39


def test_missing_arrays_and_invalid_timestamps_are_actionable(tmp_path):
    path = tmp_path / "missing.npz"
    np.savez(path, t=np.arange(3))
    with pytest.raises(ValueError, match="missing arrays"):
        summarize(path)
    write_log(path, t=np.zeros(80))
    with pytest.raises(ValueError, match="strictly increasing"):
        summarize(path)


def test_stationary_joint_does_not_invent_a_slope(tmp_path):
    path = write_log(tmp_path / "still.npz", q=np.zeros((80, 6)))
    joint = summarize(path)["active_joint_diagnostics"][0]
    assert joint["feedback_vs_angle_slope_nm_per_rad"] is None
    assert joint["feedback_vs_angle_correlation"] is None
