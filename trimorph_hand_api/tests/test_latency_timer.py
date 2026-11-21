"""Tests for the USB latency timer setup used on every DynamixelClient.connect()."""

from pathlib import Path

import pytest

from midas_hand_api.actuators import dynamixel_client as dc


def _make_sysfs(tmp_path: Path, value: int) -> tuple[Path, Path, str]:
    sysfs_root = tmp_path / "usb-serial"
    timer = sysfs_root / "ttyUSB0" / "latency_timer"
    timer.parent.mkdir(parents=True)
    timer.write_text(f"{value}\n", encoding="utf-8")
    return sysfs_root, timer, "/dev/ttyUSB0"


class FakeSerial:
    """Mimics pyserial: the low-latency flag makes the driver report 1 ms."""

    def __init__(self, timer: Path, *, fail: bool = False) -> None:
        self.timer = timer
        self.fail = fail
        self.calls: list[bool] = []

    def set_low_latency_mode(self, enabled: bool) -> None:
        self.calls.append(enabled)
        if self.fail:
            raise ValueError("Failed to update ASYNC_LOW_LATENCY flag")
        # open() rather than Path.write_text, which some tests patch to
        # emulate the root-owned sysfs file.
        with open(self.timer, "w", encoding="utf-8") as stream:
            stream.write("1\n")


@pytest.fixture(autouse=True)
def _no_sudo(monkeypatch):
    monkeypatch.setattr(dc, "_can_prompt_for_sudo", lambda: False)


def test_low_latency_flag_used_before_sysfs_write(tmp_path, monkeypatch):
    sysfs_root, timer, port = _make_sysfs(tmp_path, 16)
    serial_port = FakeSerial(timer)

    def fail_write(*_args, **_kwargs):  # the sysfs file is root-owned
        raise AssertionError("sysfs write should not be attempted")

    monkeypatch.setattr(Path, "write_text", fail_write, raising=True)
    dc._set_low_latency_timer(port, sysfs_root=sysfs_root, serial_port=serial_port)

    assert serial_port.calls == [True]
    assert timer.read_text().strip() == "1"


def test_already_low_latency_is_left_alone(tmp_path):
    sysfs_root, timer, port = _make_sysfs(tmp_path, 1)
    serial_port = FakeSerial(timer)

    dc._set_low_latency_timer(port, sysfs_root=sysfs_root, serial_port=serial_port)

    assert serial_port.calls == []


def test_flag_failure_falls_back_to_sysfs_write(tmp_path):
    sysfs_root, timer, port = _make_sysfs(tmp_path, 16)
    serial_port = FakeSerial(timer, fail=True)

    dc._set_low_latency_timer(port, sysfs_root=sysfs_root, serial_port=serial_port)

    assert serial_port.calls == [True]
    assert timer.read_text().strip() == "1"  # tmp file is writable


def test_flag_without_effect_falls_back_and_warns(tmp_path, monkeypatch, caplog):
    sysfs_root, timer, port = _make_sysfs(tmp_path, 16)

    class IgnoredFlagSerial:
        def set_low_latency_mode(self, enabled: bool) -> None:
            pass  # driver ignores the flag; timer stays at 16 ms

    def denied(*_args, **_kwargs):
        raise PermissionError("root-owned")

    monkeypatch.setattr(Path, "write_text", denied, raising=True)
    with caplog.at_level("WARNING"):
        dc._set_low_latency_timer(
            port, sysfs_root=sysfs_root, serial_port=IgnoredFlagSerial()
        )

    assert timer.read_text().strip() == "16"
    assert "could not set" in caplog.text


def test_serial_without_low_latency_support_is_skipped(tmp_path):
    sysfs_root, timer, port = _make_sysfs(tmp_path, 16)

    dc._set_low_latency_timer(port, sysfs_root=sysfs_root, serial_port=object())

    assert timer.read_text().strip() == "1"  # fell back to the sysfs write
