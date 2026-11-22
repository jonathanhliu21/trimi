"""Live per-joint plot for the teleoperation, drawn in a separate process.

The control loop calls LivePlot.push() every cycle; it only puts the sample on
a bounded queue and drops it if the plot falls behind, so drawing never slows
the loop. Each joint gets a panel with its value over the last few seconds
(follower-minus-leader angle, or follower current), red where the joint is in
contact, orange where its leader joint is only held because its finger is, and
dashed lines at +-limit.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from typing import Mapping, Sequence

import numpy as np

# Panel rows: thumb, then index / middle / ring as PIP, MCP, ABAD.
PANEL_ROWS = ((0, 1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
QUEUE_SIZE = 2000
MAX_RATE_HZ = 250.0


def panel_layout(motor_ids: Sequence[int]) -> dict[int, tuple[int, int]]:
    """(row, column) of each motor's panel: TriMI rows if they fit, else a 4-wide grid."""

    motor_ids = [int(m) for m in motor_ids]
    if set(motor_ids) <= {m for row in PANEL_ROWS for m in row}:
        rows = [row for row in PANEL_ROWS if set(row) & set(motor_ids)]
        return {m: (r, c) for r, row in enumerate(rows) for c, m in enumerate(row) if m in motor_ids}
    return {m: divmod(i, 4) for i, m in enumerate(motor_ids)}


class LivePlot:
    def __init__(self, motor_ids: Sequence[int], labels: Mapping[int, str], *, limit: float,
                 unit: str = "mA", title: str = "TriMI follower", window_s: float = 10.0):
        context = mp.get_context("spawn")  # a fresh interpreter: no serial ports or Tk state inherited
        self.queue = context.Queue(maxsize=QUEUE_SIZE)
        # If the window is closed nobody reads the queue; never wait on it at exit.
        self.queue.cancel_join_thread()
        self._pushes = 0
        self.process = context.Process(
            target=plot_main,
            args=(self.queue, [int(m) for m in motor_ids], {int(k): str(v) for k, v in labels.items()},
                  float(limit), str(unit), str(title), float(window_s)),
            daemon=True,
        )
        self.process.start()

    def push(self, t: float, current, feedback, held=None) -> None:
        """`feedback`: nonzero where the joint is in contact; `held`: true where its leader joint is held."""

        self._pushes += 1
        if self._pushes % 50 == 0 and not self.process.is_alive():
            return  # the window was closed
        state = (np.asarray(feedback) != 0).astype(np.int8) * 2  # 2 contact, 1 held, 0 free
        if held is not None:
            state = np.maximum(state, np.asarray(held, dtype=bool).astype(np.int8))
        try:
            self.queue.put_nowait((float(t), np.asarray(current, dtype=np.float32), state))
        except queue.Full:
            pass

    def close(self) -> None:
        if self.process.is_alive():
            try:
                self.queue.put_nowait(None)
            except queue.Full:
                pass
        self.process.join(timeout=1.0)
        if self.process.is_alive():
            self.process.terminate()


def plot_main(samples, motor_ids, labels, limit_value, unit, title, window_s) -> None:
    import collections

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    layout = panel_layout(motor_ids)
    rows = 1 + max(r for r, _ in layout.values())
    cols = 1 + max(c for _, c in layout.values())
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.0 * rows), sharex=True, squeeze=False)
    fig.canvas.manager.set_window_title(title)
    limit = 1.15 * limit_value
    lines, held_lines, contact_lines = [], [], []
    for ax in axes.flat:
        ax.set_visible(False)
    for motor in motor_ids:
        r, c = layout[motor]
        ax = axes[r][c]
        ax.set_visible(True)
        ax.set_title(f"{motor}: {labels.get(motor, f'motor {motor}')}", fontsize=9)
        ax.set_xlim(-window_s, 0.0)
        ax.set_ylim(-limit, limit)
        for y in (-limit_value, limit_value):
            ax.axhline(y, color="0.6", lw=0.8, ls="--")
        ax.axhline(0.0, color="0.85", lw=0.8)
        ax.tick_params(labelsize=7)
        lines.append(ax.plot([], [], lw=1.0, color="tab:blue")[0])
        held_lines.append(ax.plot([], [], lw=2.0, color="tab:orange")[0])
        contact_lines.append(ax.plot([], [], lw=2.0, color="tab:red")[0])
    for ax in axes[-1]:
        ax.set_xlabel("s", fontsize=8)
    for row in axes:
        row[0].set_ylabel(unit, fontsize=8)
    fig.tight_layout()

    keep = int(window_s * MAX_RATE_HZ)
    times = collections.deque(maxlen=keep)
    currents = collections.deque(maxlen=keep)
    contacts = collections.deque(maxlen=keep)

    def update(_frame):
        while True:
            try:
                sample = samples.get_nowait()
            except queue.Empty:
                break
            if sample is None:
                plt.close(fig)
                return lines + held_lines + contact_lines
            t, current, contact = sample
            times.append(t)
            currents.append(current)
            contacts.append(contact)
        if times:
            t = np.asarray(times) - times[-1]
            current = np.asarray(currents)
            contact = np.asarray(contacts)
            for j, (line, held_line, contact_line) in enumerate(zip(lines, held_lines, contact_lines)):
                line.set_data(t, current[:, j])
                held_line.set_data(t, np.where(contact[:, j] == 1, current[:, j], np.nan))
                contact_line.set_data(t, np.where(contact[:, j] == 2, current[:, j], np.nan))
        return lines + held_lines + contact_lines

    fig._animation = FuncAnimation(fig, update, interval=50, blit=True, cache_frame_data=False)
    plt.show()
