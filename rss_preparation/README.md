# RSS preparation: hand record → replay → overlay plots

A hand-only experiment for the TriMi wearable hand:

1. Record a trajectory with torque **off** while a person wears the hand.
2. Replay it with torque on under three conditions: `idle` (empty), `tpu`
   (TPU pseudo-hand inside), `human` (the wearer's relaxed hand inside).
   Every tick logs each motor's present current, commanded position and
   present position.
3. Overlay the three runs in two figures (current; commanded vs measured
   position), one subplot per joint, plus a per-joint summary table.

The scripts use only `midas_hand_api` (`../trimorph_hand_api`) plus numpy,
matplotlib and pyyaml. They do not need the arm, cameras or the rest of
`trimorph-piper-control`.

## Setup

From the `trimi/` root (the same environment as the top-level README):

```bash
cd trimorph-piper-control
python3 -m venv .venv
.venv/bin/pip install -e "../trimorph_hand_api[graph,test]"
source .venv/bin/activate
cd ../rss_preparation
```

## Hands

Each hand has its own YAML in `../trimorph-piper-control/config/`:

| Hand | Config | USB adapter | Calibration | Motors |
|---|---|---|---|---|
| Trimi1 | `trimi1.yaml` | FTBEO9P0 | first-run hard stops, 2026-11-20 | XC330-M181-T on 2, 3, 5, 8, 11; XL330-M077-T elsewhere |
| Trimi2 | `trimi2.yaml` | FTBEO3G5 | not homed (offsets 0) | XL330-M077-T on 0, 4, 7, 10; XC330-M288-T on 2; XC330-M181-T elsewhere |

Both open the joint limits to ±100 rad for this test.

- **Picking the hand.** Each YAML pins its adapter in `port`. With no
  `--config`, the scripts use whichever known hand is plugged in. If neither
  or both are plugged in, they ask for `--config`. `--port` overrides the
  pinned port.
- **Motor check.** `motor_models` in each YAML lists the model on every ID.
  The scripts check it on connect, so running one hand's config on the other
  fails with the mismatched IDs. midas_hand_api itself takes a single
  `motor_model` per hand; XL330 and XC330 share the control table and units,
  so `xl330-m077-t` works for these mixed hands.
- **After swapping a motor:**
  1. Set the new motor's ID and baud rate (1 Mbps).
  2. Update `motor_models`.
  3. Re-record. A new motor has a different zero, so older recordings no
     longer line up. Replay refuses a recording whose saved motor models
     differ from the hand. This only works for sessions recorded after
     2026-11-22; older recordings don't save their motor models.
- **Adding a hand.** Write its YAML with `port` and `motor_models`, then
  append it to `HAND_CONFIGS` in `rss_common.py`.

Close Dynamixel Wizard (or at least disconnect it) before running the
scripts. Two programs on the bus garble each other's packets.

## Workflow

```bash
python smoke_test.py                                  # read-only status table + read-rate benchmark
python record_trajectory.py --session grasp01 --duration 15
python replay_trajectory.py grasp01 --condition idle
python replay_trajectory.py grasp01 --condition tpu
python replay_trajectory.py grasp01 --condition human
python plot_trials.py grasp01
```

Each session lives in `data/<session>/` (gitignored):

```
trajectory/leader_hand.csv, metadata.json     # desired trajectory
replays/{idle,tpu,human}.csv (+ .json)         # telemetry + settings/diagnostics
figures/current_overlay.{png,pdf}
figures/position_overlay.{png,pdf}
figures/summary.csv                            # per joint x condition: RMS/max error, |current| stats
```

The CSV formats match the existing tools. `leader_hand.csv` uses the columns
of `collect_hand_only.py`, so `data_collection/replay_episode.py` can replay it.
The replay CSVs use `ReplayCurrentRecorder`'s schema, so
`scripts/cli/gen_graph.sh` and `gen_graphs_from_csvs.py` can read them.

### Script notes

- **smoke_test.py** pings all 13 motors. It reports model, firmware, Return
  Delay Time, mode, torque, decoded hardware errors, voltage, temperature,
  position (and whether it is within the YAML limits) and current. It then
  times 200 sync reads. It exits non-zero on missing motors, hardware errors,
  temperature above 60 °C, voltage outside 4.5–5.8 V, or a USB latency timer
  above 1 ms. Opt-in actions:
  - `--motion-test` moves each motor ±0.15 rad and back in mode 5 with a
    300 mA cap. The hand must be empty.
  - `--set-return-delay-zero` writes Return Delay Time = 0 to motor EEPROM.
- **record_trajectory.py** first switches the motors to the replay's mode 5,
  with torque still off, so no mode change happens between recording and
  replay. Swapped or reset motors come up in factory mode 3. It then records
  at 100 Hz by default. Press Enter or pass `--duration` to stop. It reports
  the achieved rate, read errors and range of motion.
- **replay_trajectory.py** runs once per condition, at 100 Hz by default:
  1. Checks the hand's motor models against the recording's, then prompts
     with torque still off, so you can insert the TPU or your hand.
  2. Configures the mode, current cap and gains, and sets the motion profile.
  3. Enables torque, holding the current pose.
  4. Runs a 3 s cosine ramp to the first sample and settles for 1 s.
  5. Replays, logging every tick.
  6. Holds the last pose for 1 s, then turns torque off.

  Controller defaults:
  - **Motion profile:** 6 rad/s, 50 rad/s² (the policy rollout in
    `executor.py` uses 3 rad/s). Motors never exceed 6 rad/s, so faster recorded motion
    lags; `record_trajectory.py` flags joints above that speed.
    `--profile-velocity 0` disables it.
  - **Gains:** P/I/D 2000/0/400. I is 0 here (100 in the YAML and the
    rollout) to avoid integral windup when current saturates. Override with
    `--p-gain`, `--i-gain`, `--d-gain`.
  - **Smoothing:** the recording gets a zero-phase 5 Hz Gaussian low-pass
    before replay. It never overshoots the recorded range; the smoothed
    trajectory is what gets commanded and logged. `--smooth-hz 0` disables it.

  Every setting is saved in `replays/<condition>.json` and shown in the
  figure notes. Use the same settings for all three conditions. Use
  `--playback-speed 0.5` for half speed. Reruns need `--force`.
- **plot_trials.py** auto-discovers `replays/{idle,tpu,human}.csv` (and any
  other `*.csv` there). It plots current multiplied by the joint sign, so
  positive current drives positive joint motion; `--raw-current-sign`
  disables that. `--degrees` plots positions in degrees. For runs outside a
  session, use `--csv LABEL=PATH --out DIR`.

## Safety and data caveats

- **Mode 5 with a current cap (default).** Goal Current (the YAML's 1700 mA,
  or `--current-limit-ma`) caps torque in hardware. A trace that flat-tops at
  the cap was saturated. Use the same cap for all three conditions so they are
  comparable. `--operating-mode 3` removes the hardware cap.
- **Software abort.** If any |current| stays at or above `--abort-current-ma`
  (default 0.95 × cap) for `--abort-samples` ticks (default 10 = 0.1 s),
  torque turns off immediately. The partial run is saved and flagged.
  Ctrl-C and any error also turn torque off.
- **No joint-limit clipping by default.** The recorded positions were
  physically produced by the wearer, so they are reachable. Some YAML
  limits/offsets do not match this hand right now: Thumb IP, Thumb CMC roll
  and Ring abduct read outside their limits at rest. Clipping would therefore
  jerk those joints to a limit. `--clip-to-limits` restores clipping.
- **Whole-turn offsets after a power cycle.** In mode 5, positions are
  multi-turn, but a power cycle resets each motor to a single turn. A motor
  near the 0/4096 boundary (IDs 0 and 1 currently) then reads exactly 2π away
  from its recorded value. The replay removes whole turns automatically
  (identical physical pose) and refuses to start if any joint is still more
  than `--max-start-delta-rad` (1.5) from the first sample.
- **Absolute angles depend on the calibration.** Tracking error and current
  are unaffected by the offsets above. `trimi2.yaml` is not homed, so its
  angles are raw motor angles times the joint sign. For anatomically
  meaningful absolute angles in the figures, re-home the hand before
  recording (see `../trimorph_hand_api/README.md`).

## USB latency

FTDI adapters default to a 16 ms latency timer, which caps every sync read at
16 ms (about 62 Hz). `midas_hand_api` now sets the tty's `ASYNC_LOW_LATENCY`
flag on connect. The Linux ftdi_sio driver then uses 1 ms, with no sudo needed
(being in `dialout` is enough). On this machine a 13-motor
position/velocity/current read went from 16.0 ms to 3.0 ms (about 333 Hz).
`smoke_test.py` prints the timer value. If it stays above 1 ms, install the
persistent udev rule:
`sudo ../trimorph_hand_api/setup_dynamixel_latency.sh <by-id port>`.

## Tests

```bash
python -m pytest tests        # offline, uses a fake hand; no hardware needed
```
