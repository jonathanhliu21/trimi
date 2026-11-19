# TRIMORPH + MIDAS/PiperX control

This repository contains hardware bring-up, contact-model testing, TRIMORPH-only
data collection, TRIMORPH-to-MIDAS retargeting and force feedback, episode
replay, and PiperX/OAK-1 collection tools.

Run commands from the repository root. Replace every example serial device with
the stable `/dev/serial/by-id/...` path for the corresponding hand.

## Safety and configuration

- TRIMORPH uses 13 XL330-M077-T motors with IDs `0..12`. During normal
  collection it is backdriven with torque disabled; force-feedback collection
  temporarily places selected leader joints in current mode.
- MIDAS is the commanded follower. Verify its calibration, limits, gains,
  current limits, motor order, and serial port before enabling torque.
- Motion tools refuse unreviewed YAML files. Check every calibration value and
  set `placeholders_unreviewed: false` only after review.
- Start motion tests at a conservative current limit with an unobstructed hand
  and access to power/E-stop controls. Options such as `--yes` and
  `--skip-target-safety` bypass safeguards and should only be used deliberately.

The two principal checked-in configurations are:

| Configuration | Used by |
| --- | --- |
| `config/xl330_test_config.yaml` | TRIMORPH homing, angle recording, deterministic motion, current logging, and contact tests in `joint_test.py` |
| `config/midas_hand_config.yaml` | MIDAS contact UI, follower retargeting, replay onto MIDAS, and thumb calibration |

`data_collection/collect_hand_only.py` and direct TRIMORPH replay default to
`config/hand.yaml`. Create that local, calibrated runtime file from
`config/hand.example.yaml`, or pass `--hand-config` explicitly:

```bash
cp config/hand.example.yaml config/hand.yaml
```

## Project structure

| Location | Contents |
| --- | --- |
| `config/` | Hand YAML configurations, including `xl330_test_config.yaml` |
| `data_collection/` | `collect_hand_only.py`, `replay_episode.py`, and collection documentation |
| `data_collection/collection/` | Shared runtime configuration, episode storage, workers, contact detection, and retargeting |
| `scripts/thumb_retargeting/` | Paired thumb calibration and model fitting |
| `scripts/sim/` | TriMI MuJoCo model builder and the 1:1 hand-to-sim mirror |
| `scripts/teleop/` | TriMI-to-TriMI 1:1 joint teleoperation and Quest-to-Piper teleoperation |
| `scripts/motion_smoothness/` | Deterministic joint motion tests |
| `scripts/tracking_error/` | Leader/follower tracking diagnostics |
| `scripts/dataset/` | Piper-only postprocessing base and dataset validation (`data_collection/postprocess_dataset.py` extends it) |
| `scripts/trimorph_api_test/` | Hand bring-up, contact UI, telemetry, and plotting |
| `scripts/piper_api/`, `scripts/camera/`, `scripts/quest/` | Arm utilities and standalone sensor checks |
| `tests/` | Automated tests |

Collection and replay share `HandRuntimeConfig` in
`data_collection/collection/config.py`, including hand profiles, retarget scales
and offsets, camera defaults, dataset location, and Piper connection defaults.
`HandOnlyCollectionConfig` extends it with worker and episode options.
Replay-only action multipliers, rates, and preposition settings live in
`data_collection/collection/replay_config.py`. The Piper SDK is loaded when an
arm connection is created, so hand-only tools can run without it.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../trimorph_hand_api
```

The package installed from `trimorph_hand_api` is imported in this repository
as `midas_hand_api`. The core Python dependencies are NumPy, PyYAML (provided by
the hand API), and the hand API itself. Matplotlib is required for graph
generation. OpenCV and DepthAI are needed by the full camera collection
pipeline. Tkinter is required by the contact UIs and is normally supplied by
the system Python installation. The PiperX `pyAgxArm` package is installed
directly from the vendor's GitHub repository by `requirements.txt`; configure
SocketCAN before using PiperX.

Bring up the default `can0` interface at 1 Mbit/s with the checked-in helper:

```bash
./setup_arm_can.sh
```

Use `./setup_arm_can.sh --help` to select another interface or bitrate. The
script uses `sudo` for the link changes and prints the resulting CAN status.

Optional Paxini tactile operation also requires the Paxini support exposed by
`midas_hand_api` and a valid sensor serial port.

## 1. TRIMORPH finger bring-up and deterministic current tests

[`scripts/trimorph_api_test/joint_test.py`](scripts/trimorph_api_test/joint_test.py)
is the main bring-up tool. The global `--config` and `--test-finger` options go
before the subcommand. Finger choices are `thumb`, `finger-1`, `finger-2`,
`finger-3`, and `all`; non-homing commands default to `finger-1`.

Home one finger, or selected joints within that finger:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config config/xl330_test_config.yaml \
  --test-finger finger-1 \
  home --current-threshold-ma 200 --velocity-rad-s 1.5

python3 scripts/trimorph_api_test/joint_test.py \
  --config config/xl330_test_config.yaml \
  --test-finger thumb \
  home --joints 0 2
```

Omit `--test-finger` and `--joints` to home every configured motor. Homing
updates the selected motors' offsets while preserving the other saved offsets.

Read calibrated joint angles with torque disabled and write them to CSV:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config config/xl330_test_config.yaml \
  --test-finger finger-1 \
  record --motors 4,5,6 --rate-hz 20 --duration-s 30 \
  --print-every 10 --output output/finger_1_angles.csv
```

Run a deterministic open-close-open path and capture graph-compatible current,
target/actual position, and target/actual velocity telemetry. `--duration-s` is
the duration of each leg, and `--trials` makes repeated trials available for
averaging:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config config/xl330_test_config.yaml \
  --test-finger finger-1 \
  stream_move --motors 4,5,6 --rate-hz 50 --duration-s 2 \
  --settle-s 0.5 --trials 5 \
  --output-current output/finger_1_stream.csv
```

For discrete configured-profile moves instead, use:

```bash
python3 scripts/trimorph_api_test/joint_test.py \
  --config config/xl330_test_config.yaml \
  --test-finger finger-1 \
  profile_move --motors 4,5,6 \
  --profile-velocity-rad-s 1.5 \
  --profile-acceleration-rad-s2 10 --trials 5 \
  --output-current output/finger_1_profile.csv
```

`profile_move_sequential` checks the configured motor mapping one motor at a
time. `measure_current`, `measure_position_error`, and `detect_contact` provide
continuous console measurements at selected targets. Run `python3
scripts/trimorph_api_test/joint_test.py COMMAND --help` for their complete
options. Existing `--output-current` files are protected unless `--force` is
passed.

## 2. MIDAS contact-model testing UI

[`scripts/trimorph_api_test/hand_ui.py`](scripts/trimorph_api_test/hand_ui.py)
opens a MIDAS controller and live contact visualizer:

```bash
python3 scripts/trimorph_api_test/hand_ui.py
```

The UI defaults to `config/midas_hand_config.yaml`; choose another reviewed
configuration in its **Config** field if needed. After connecting, use the
joint sliders or the finger/thumb open-close-open buttons. The UI displays
position/current-based contact state and exposes the motion-current threshold,
velocity threshold, settle/persistence timing, and contact feedback threshold.

To include Paxini fingertip contact sensing and a diagnostic log:

```bash
python3 scripts/trimorph_api_test/hand_ui.py \
  --add-tactile \
  --paxini-port /dev/serial/by-id/PAXINI \
  --tactile-debug-log output/paxini_contact.jsonl
```

`--paxini-port` or `--tactile-debug-log` also enables tactile sensing, so
`--add-tactile` is optional when either is supplied.

## 3. Graph generation from joint tests and replay

[`scripts/trimorph_api_test/gen_graphs_from_csvs.py`](scripts/trimorph_api_test/gen_graphs_from_csvs.py)
accepts the telemetry schema written by both `joint_test.py --output-current`
and `replay_episode.py --output-current`. The angle-only CSV produced by
`joint_test.py record` is not an input to this graph tool.

Generate an overview of current, commanded/measured position, and
commanded/measured velocity from a deterministic joint test:

```bash
python3 scripts/trimorph_api_test/gen_graphs_from_csvs.py \
  output/finger_1_stream.csv \
  --combine-mode average \
  --joints 4 5 6 \
  --titles "Finger 1 stream" \
  --output output/finger_1_stream.png
```

`--titles` requires exactly one title per input CSV. Compare joint-test and
replay telemetry directly with:

```bash
python3 scripts/trimorph_api_test/gen_graphs_from_csvs.py \
  output/finger_1_stream.csv output/replay_current.csv \
  --titles "Bench path" "Episode replay" \
  --overlay --plot-finger --combine-mode sequential \
  --joints 4 5 6 \
  --output output/bench_vs_replay.png
```

Use `--combine-mode average` for an interpolated mean and ±1 standard deviation
across trials, or `sequential` to concatenate trials in time. `--plot-finger`
uses finger-specific rows; without it the output is the three-panel overview.

## 4. Collect from TRIMORPH only

[`data_collection/collect_hand_only.py`](data_collection/collect_hand_only.py)
can record the backdriven TRIMORPH without launching PiperX. The wrist camera
is optional. Collection is foot-pedal controlled by default: one press starts
an episode after the ready delay, and pressing and releasing again saves it.
Holding the pedal for the displayed delete interval discards the active
episode. The pedal is auto-detected from Linux `/dev/input/event*` devices and
is exclusively grabbed while the collector runs so its keystroke is not sent
to other applications. If detection fails, run `python3
scripts/piper_api/record_trajectory.py --list-input-devices` and ensure the
desired event device is readable by the current user.

For unattended or fixed-duration collection, pass `--episode-s` (the older
`--episode-time` spelling remains supported):

```bash
python3 data_collection/collect_hand_only.py \
  --data-collection \
  --dataset-root data_trimorph \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --active-motors 0 1 2 3 4 5 6 7 8 9 10 11 12 \
  --warmup-time 5 --episode-s 30
```

This example warms up, records one 30-second episode, and exits. Without
`--episode-s`, the collector continues creating pedal-controlled episodes until
`q`, Escape (when a camera preview is open), or Ctrl-C. A hand-only episode
contains populated `leader_hand.csv` and `metadata.json`, plus empty
`camera_timestamps.csv` and `piper.csv` bookkeeping files. `follower.csv` is
written only when a follower is configured, and no camera video is created.
The older filename `proprioception.csv` is still accepted by replay for
compatibility.

Add `--use-piper` to passively record the Piper EEF pose and drag-teach state
to `piper.csv`, synchronized with the hand samples. This uses the shared Piper
settings and passive logger; it does not enable or send motion commands to the arm. Move the arm with its built-in drag-teach control.
Add `--joint-level` to include measured `joint_1`–`joint_6` in each Piper row,
then use `--joint-level` with `data_collection/replay_episode.py` to replay
those angles using smooth joint commands instead of firmware EEF IK. This only
works for episodes newly recorded with joint columns. `--joint-position` is an
alias for collection. While recording joint angles, the Piper worker warns
within 2° of a controller limit or when a measured angle is beyond it; it does
not interrupt recording. Episode joint replay shares the trajectory replayer's
bounded-step control and accepts the same `--max-joint-step-rad`,
`--preposition-seconds`, `--settle-seconds`,
`--preposition-speed-percent`, and `--replay-speed-percent` tuning options.
Edit `JOINT_ACTION_SCALES` in `data_collection/collection/replay_config.py` to set the
final-position multiplier for each hand motor ID 0–12.
Piper sampling defaults
to 250 Hz; `--piper-rate-hz`, `--piper-teacher-mode-poll-hz`,
`--piper-channel`, `--piper-interface`, `--piper-firmware`,
`--piper-installation-pos`, and `--piper-payload` configure the Piper worker.

Add `--add-camera` to start the DepthAI/OAK wrist camera, display its live feed,
and save `camera.mp4` plus populated `camera_timestamps.csv` and camera metadata
in the standard raw episode format. Press `q` or Escape in the camera
window to stop through the normal episode-save path. Auto-exposure and auto
white balance are off: the camera uses the fixed `TASK_CAMERA_EXPOSURE_US`,
`TASK_CAMERA_ISO`, `TASK_CAMERA_WHITE_BALANCE_K` and `TASK_CAMERA_FPS` in
`data_collection/collection/config.py`. Re-measure them if the lighting changes.
Set `TASK_CAMERA_FLIP = "180"` in the same config file to rotate the saved and
previewed image by 180 degrees.

To check the camera and its configured exposure, white balance, resolution,
and flip before collection, run:

```bash
python3 scripts/camera/test_wrist_camera.py
```

## 5. TRIMORPH leader to MIDAS follower with force feedback

Without a follower configured, all active TRIMORPH motors are automatically
torque-disabled while their positions are recorded, including thumb joints 2–3.
This also applies when tactile sensing is enabled without a follower.

Provide both follower options to enable retargeting. `--retarget-only` runs
until Ctrl-C without saving an episode; omit it (or use `--data-collection`) to
record one fixed-duration episode:

```bash
python3 data_collection/collect_hand_only.py \
  --retarget-only \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --follower-config config/midas_hand_config.yaml \
  --follower-port /dev/serial/by-id/MIDAS \
  --thumb-model thumb_retarget_model.json \
  --show-ui
```

TRIMORPH positions are scaled/offset, the learned thumb model replaces MIDAS
joints 2-3, and targets are clipped and smoothed before being sent to MIDAS.
The follower's detected contact error is sent back to selected TRIMORPH joints
as capped current-mode force feedback. `--show-ui` requires a follower. Add
`--add-tactile --paxini-port /dev/serial/by-id/PAXINI` to combine Paxini
fingertip contacts with force feedback, or use `--tactile-debug-log PATH` for
headless tactile diagnostics.

Use `--disable-force-feedback-motors` to leave selected TRIMORPH motors
torque-disabled without removing them from position reading or retargeting. The
exclusion applies to both MIDAS contact feedback and Paxini tactile feedback:

```bash
python3 data_collection/collect_hand_only.py \
  --retarget-only \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --follower-config config/midas_hand_config.yaml \
  --follower-port /dev/serial/by-id/MIDAS \
  --disable-force-feedback-motors 8
```

## 5b. TriMI leader to TriMI follower (1:1 joints)

Two identical TriMI hands, each on its own adapter (both use motor IDs 0-12,
so they cannot share a bus). Home each hand into its own calibration, the
follower from the leader's file so both use the same homing table and read
the same angle in the same pose. Take the hands off and keep the fingers
clear while homing:

```bash
TRIMI_LIVE=~/TriMI-Copilot/.venv/bin/trimi-live
$TRIMI_LIVE home --hand-port /dev/serial/by-id/<LEADER> \
  --out ~/.config/trimi-copilot/hand_leader.yaml
$TRIMI_LIVE home --hand-config ~/.config/trimi-copilot/hand_leader.yaml \
  --hand-port /dev/serial/by-id/<FOLLOWER> --out ~/.config/trimi-copilot/hand_follower.yaml
```

Then teleoperate. The leader is unpowered and moved by hand; its calibrated
joint angles go unchanged to the follower:

```bash
python3 scripts/teleop/trimi_teleop.py \
  --leader-port /dev/serial/by-id/<LEADER> \
  --follower-port /dev/serial/by-id/<FOLLOWER> --print
```

The follower eases onto the leader's pose over `--ramp-s` (2 s), then follows
at `--rate-hz` (100). Its goals are clipped to its joint limits, its current is
capped at `--current-limit-ma` (400 mA) and its speed at `--max-speed-rad-s`
(6). The script refuses calibrations with different motor IDs, joint signs or
homing tables, a follower file that was never homed, and joints reading far
outside their limits (a power cycle can shift a joint by a turn; home again).
A leader reading that jumps more than `--max-jump-rad` in one cycle stops it,
and both hands' torque is off on exit.

`--force-feedback`: a follower joint is in contact when, for 20 ms, it is still
(under `--contact-still-rad-s`, 0.25; a slow follower still tracking moves at
0.2-0.6) and the leader is more than `--contact-error-rad` (0.04) past it and
still pulling away from it. A gap that stays put is the follower falling short
of its goal, not an object; one past twice the contact error counts anyway. The motor current does not count by default: it mixes gear ratio, moment
arm, acceleration and friction, differs from finger to finger, and cannot tell
a fast move from a light touch. `--contact-current-ma N` adds the condition
that the follower also pushes towards its goal with more than N mA. A follower
within 0.1 rad of the stop homing found for that joint is at its own stop, not
on an object (the leader meets its own stop there too). `--grasp-only` makes
the index/middle/ring PIP and MCP (flexion reads negative) count only errors in
the grasping direction; add `MOTOR=+1`/`MOTOR=-1` for others, e.g.
`--grasp-only 1=-1` for a thumb joint. The other joints count both directions.
`--contact-motors` (`contact_motors: [0, 2, 4, 5, 7, 8, 10, 11]`) limits
contact detection to some joints; the others are still walled with their
finger. `--contact-motor-overrides MOTOR:error_rad=E,hold_ms=H`
(`contact_motor_overrides: {0: {error_rad: 0.06}}`) gives single joints their
own contact error and hold time: thumb joint 0 is geared high and lags at the
start of a fast move, long enough to pass for contact at the common 0.02 rad.

At contact the leader is stopped where the follower stopped (a wall): the
joint's own motor holds the leader at the follower's position at contact, in
current-based position mode with up to `--wall-current-ma` (700; the XL330
allows 1750), its internal loop far stiffer than the 100 Hz teleop loop could
render. The wall stays there (`--wall-follow` moves it with the follower, so a
soft object gives, but so do the follower's flex and gear play), and the
follower keeps following the leader, so pressing harder bends the leader
slightly past the wall and the follower squeezes harder. For the first
`--wall-settle-ms` (120) the wall holds whatever the hand does, so the leader is
brought back to where the follower stopped, at `--wall-speed-rad-s` (3); only
the follower moving on freely ends it sooner. After that the wall lets go the
moment the hand opens: the leader comes `--wall-release-rad` (0.01) out of it,
or its motor current turns to pull the finger in by `--wall-release-ma` (40).
At any time it lets go when the follower moves on past it by `--wall-pass-rad`
(0.03) or faster than the still speed: nothing was there. `--wall-p-gain`/
`-i-gain`/`-d-gain` set its stiffness (default: the leader calibration's), and
`--wall-motor-gains MOTOR:p=P,i=I,d=D` (`wall_motor_gains: {0: {p: 600, i: 0}}`)
one joint's over them: a joint geared too high to give under the hand (thumb
joint 0) gets a softer P, and no I, which would wind up against a steady push
until the wall pushes back with full current.

Walls go up a whole finger at a time (thumb 0-3, index 4-6, middle 7-9, ring
10-12): while any joint of a finger is in contact, its other joints are held
too, each at its follower's position, and the hand opening at any of them
takes the whole finger's walls down. For a held joint not in contact itself,
opening is judged in its grasping direction, so the PIP/MCP under
`--grasp-only` can release the finger but an ABAD joint cannot.
`--no-wall-whole-finger` (`wall_whole_finger: false`) walls each joint on its
own. Out of contact the leader's joints are unpowered (torque off), since a
powered Dynamixel drags when turned by hand; on exit they are left torque off
with the calibration's current cap and gains.

`--log` writes the file when the teleop stops, also on a second Ctrl-C,
SIGTERM, a closed terminal (SIGHUP) or a failed shutdown; only SIGKILL loses it.

With `--print` each line also shows the loop rate and the bus time per cycle.
The two hands are read at the same time on their own buses, so a cycle takes
about as long as the slower read. `--plot` opens a live plot of each joint's
follower-minus-leader angle (`--plot current` for the follower's current), red
while in contact and orange while held only because the rest of its finger is,
in a separate process that never slows the loop. `--print` lists both: the
joints in contact and the leader joints walled.

Every tunable has a long option and a key in `config/trimi_teleop_tuning.yaml`,
which is read on every run: built-in defaults, then that file (or `--tuning
FILE`, or none with `--no-tuning`), then the command line. The follower's
position P/I/D gains can be overridden there too (`follower_p_gain`, ...).
`--save-tuning FILE` writes the settings a command line amounts to and exits,
so a tuned command line can be kept as a file.

How well the follower's current alone shows contact can be measured: record a
run with free motion only and one pushing the follower on a rigid object, then
compare them:

```bash
python3 scripts/teleop/trimi_teleop.py ... --log free.npz      # move freely, touch nothing
python3 scripts/teleop/trimi_teleop.py ... --log contact.npz   # squeeze an object, light to firm
python3 scripts/teleop/analyze_teleop_log.py --free free.npz --contact contact.npz
```

Per joint it prints the free-motion current, the floor left after a fitted
friction/motion model, the contact current, mA per radian of push, and how far
past first contact the leader must push before the current clears the floor.

## 6. Episode replay and current logging

Replay the latest episode on TRIMORPH and PiperX (or supply an episode number,
`episode_XXXXXX`, or a directory path):

```bash
python3 data_collection/replay_episode.py \
  --dataset-root data_trimorph \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH
```

Hand replay always selects Dynamixel pure position-control mode (`3`) for
faithful pose tracking. This is an in-memory replay override; it does not alter
the hand YAML or the current-based position mode used during collection.

Replay just episode 42's hand stream and record graph-compatible current
telemetry at the replay logger's fixed 10 Hz sampling rate:

```bash
python3 data_collection/replay_episode.py 42 \
  --dataset-root data_trimorph \
  --hand-only \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --rate-hz 30 --preposition-seconds 3 --settle-seconds 1 \
  --output-current output/episode_42_replay_current.csv
```

To replay the recorded TRIMORPH stream as retargeted commands on MIDAS:

```bash
python3 data_collection/replay_episode.py 42 \
  --dataset-root data_trimorph \
  --hand-only --use-follower \
  --hand-config config/midas_hand_config.yaml \
  --hand-port /dev/serial/by-id/MIDAS \
  --thumb-model thumb_retarget_model.json \
  --output-current output/episode_42_midas_current.csv
```

The `ACTION_SCALE` constant in `data_collection/collection/replay_config.py` multiplies
every final hand target angle before joint-limit clipping. The
`JOINT_ACTION_SCALES` dictionary adds a multiplier by motor ID for all IDs
0–12; IDs 4–5 and 7–8 currently use `1.3`, and the others use `1.0`.
Useful switches include
`--no-preposition`, `--arm-only`, and `--no-piper-enable`.
`--hand-only` and `--arm-only` are mutually exclusive in practice, and current
logging cannot be used with `--arm-only`.

### Record and replay a Piper trajectory

[`scripts/piper_api/record_trajectory.py`](scripts/piper_api/record_trajectory.py)
records a passively drag-taught Piper end-effector trajectory. First enable the
arm's drag-teach mode, then press the pedal once to begin and again to stop and
save:

```bash
python3 scripts/piper_api/record_trajectory.py \
  --output trajectories/piper_pick.csv \
  --joint-level \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

Use `--episode-s 10` to stop automatically after ten seconds, `--force` to
replace an existing output, and `--list-input-devices` to locate the pedal.
The recorder leaves Piper passive and stores timestamped end-effector
`x,y,z,roll,pitch,yaw` samples in CSV. With `--joint-level`, it also stores
measured `joint_1`–`joint_6` angles. The default sampling request is 250 Hz;
actual distinct samples are limited by the controller's feedback rate. In
`--joint-level` mode, the recorder prints a warning when a measured joint is
within 2° of, or beyond, a controller-reported limit; recording continues.

Replay that CSV in Piper position mode with automatic movement to its first
pose followed by time-based interpolation at the requested command rate:

```bash
python3 scripts/piper_api/replay_trajectory.py \
  trajectories/piper_pick.csv \
  --joint-level --rate-hz 60
```

The replay command also accepts an episode directory containing
`trajectory.csv` or `piper.csv`. Use `--debug` for controller state and
commanded-versus-measured pose diagnostics, or `--no-preposition` only when the
arm is already safely positioned at the first pose.
Joint-level replay uses smooth `move_j` commands, not the unsmoothed `move_js`
mode. It does not preflight recorded angles against controller limits; the
controller may reject out-of-range targets during replay. Omit `--joint-level`
for the original EEF replay.
Higher sampling or command rates do not guarantee lower physical tracking lag.
For joint-level replay, prepositioning ramps linearly from the arm's measured
joint angles to the first recorded point. Each subsequent command is capped at
`--max-joint-step-rad` (default 0.02 rad) from measured feedback. If the start
is too far to reach in `--preposition-seconds`, replay aborts before that move;
increase the preposition time only after checking the physical path is clear.
After prepositioning, joint-level replay warns if the largest start error is
over 0.10 rad and aborts only above 0.15 rad.

### Alternating manual replay

[`scripts/replay_corl_workshop/manual_replay.py`](scripts/replay_corl_workshop/manual_replay.py)
replays separate Piper and TRIMORPH trajectories one at a time. Create a text
file whose non-comment lines alternate from Piper to TRIMORPH; relative paths
are resolved from the text file's directory:

```text
# workshop_sequence.txt
trajectories/piper_pick.csv
trajectories/trimorph_close.csv
trajectories/piper_place.csv
trajectories/trimorph_open.csv
```

Then run:

```bash
python3 scripts/replay_corl_workshop/manual_replay.py workshop_sequence.txt \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH \
  --foot-pedal-device /dev/input/by-id/FOOT_PEDAL
```

Both devices are checked before motion. The first fresh pedal press moves the
hand to the initial joint-angle constant defined in the script. After that move
settles, a second fresh press starts the first Piper trajectory. After each CSV
finishes, press and release the foot pedal to advance; a press made during
motion is discarded. The inactive controller stays enabled at its last target,
and no trajectory commands are sent to it until its next turn. Use
`--list-input-devices` to find the pedal event device and `--help` for
replay-rate, preposition, hardware, and motion-profile options.

## 7. Thumb IK/retarget calibration

[`scripts/thumb_retargeting/retarget_thumb.py`](scripts/thumb_retargeting/retarget_thumb.py)
collects corresponding TRIMORPH and MIDAS thumb poses for the joints 2-3
retarget map. TRIMORPH stays torque-disabled while MIDAS moves through a
deterministic, joint-limit-aware calibration set:

```bash
python3 scripts/thumb_retargeting/retarget_thumb.py \
  --trimorph-config config/hand.yaml \
  --trimorph-port /dev/serial/by-id/TRIMORPH \
  --midas-config config/midas_hand_config.yaml \
  --midas-port /dev/serial/by-id/MIDAS \
  --output thumb_tip_matching.csv \
  --num-points 40 --seed 7 --limit-margin 0.05 \
  --max-step 0.04 --command-rate 50 \
  --settle-time 0.5 --samples 10 --sample-rate 100
```

At each prompt, hold the backdriven TRIMORPH thumb at the pose corresponding to
the displayed MIDAS pose and press Enter. The CSV is checkpointed after every
point. Then fit the affine retarget model used by collection and follower
replay:

```bash
python3 scripts/thumb_retargeting/fit_model_to_endpoints.py \
  --input thumb_tip_matching.csv \
  --output thumb_retarget_model.json
```

Use `--no-configure-midas` only when the required MIDAS operating mode and gain
registers have already been configured correctly.

## Standalone hand diagnostics

```bash
python3 scripts/motion_smoothness/test_motion_smoothness.py --help
python3 scripts/tracking_error/get_tracking_error.py --help
```

Tracking diagnostics write to `output/tracking_error/` by default. Motion
smoothness tests use the local calibrated `config/one_finger_test.yaml`.

## TriMI in MuJoCo (1:1 mirror)

Build the MuJoCo model once from the Onshape URDF export (zip, folder, or
`.urdf`). It is written to `assets/trimi_v2/`, which is not tracked:

```bash
python3 scripts/sim/build_trimi_sim_model.py ../TRIMI_URDF_v2.zip
```

The export does not load into MuJoCo as is: the palm mesh exceeds MuJoCo's
200,000-face STL limit, every part was exported at 1 kg/m^3 (masses about 1000x
too small), and there is no collision geometry. The builder fixes these and
prints the joint chains it found.

Then move the hand and watch the model follow. The motors stay torque-disabled.
By default the script reads the calibration `trimi-live home` saves
(`~/.config/trimi-copilot/hand.yaml`) and the only adapter in
`/dev/serial/by-id`; pass `--hand-config` / `--hand-port` otherwise:

```bash
python3 scripts/sim/trimi_mirror.py
```

`config/trimi_v2_joint_map.yaml` gives each motor's URDF joint as
`q_urdf = sign * scale * (q_leader - zero)`. MCP and PIP use scale 34/84: a
34-tooth gear on the motor drives an 84-tooth sector on the next link. ABAD and
thumb joints are direct (scale 1).

The zeros synchronize the model with the hand at the hard stops that homing
drives into. Homing leaves each motor reading `-joint_sign * cad_offset` at its
stop; `sync_hard_stops.py` turns the same joint in the URDF until its link meets
the palm or the links it is mounted on (the motor horn, drive gears and other
fingers are ignored, as in homing) and sets the zero so both agree. The zeros
depend on the homing table, not on the offsets a homing measures, so re-run it
only when the table changes (`--reverse`, `--zero-here`, `--stop-deg`) or a
sign is flipped:

```bash
python3 scripts/sim/sync_hard_stops.py --source ../TRIMI_URDF_v2.zip --write
```

Without `--source` it uses the model's decimated meshes (within about 0.5 deg).
A joint that homes against a false stop (held or blocked) reads wrong in both
the hand's own tools and the model; re-homing it fixes both.

## Important shared modules

| Module | Role |
| --- | --- |
| `data_collection/collection/contact_detection.py` | Shared current/velocity/position-error state machine and thresholds used directly by `joint_test.py` and `hand_ui.py`, and by the MIDAS follower worker during force-feedback retargeting |
| `data_collection/collection/follower_hand_worker.py` | TRIMORPH-to-MIDAS scaling, thumb-model application, smoothing, follower contact detection, and follower telemetry |
| `data_collection/collection/leader_hand_worker.py` | Torque-disabled leader reads plus capped current-mode force feedback on TRIMORPH |
| `data_collection/collection/thumb_retargeting.py` | Loads and applies the learned joints 2-3 thumb retarget model |
| `data_collection/collection/hand_common.py` | Loads narrowed hand configs and verifies connected Dynamixel models |
| `data_collection/collection/tactile_contact_detection.py` | Paxini moving-average contact detection used by the tactile UI/worker paths |

Changing contact thresholds in `contact_detection.py` affects both standalone
contact test scripts and the force-feedback follower path; validate changes on
low-current hardware tests.

## Full PiperX/OAK-1 collection and dataset conversion

The collector records the hand, Quest, Vicon mocap, and OAK-1 camera by
default, into the next free `data_collection_n` folder. Add `--use-piper` to
record PiperX drag-teach poses instead of the Quest:

```bash
python3 data_collection/collect_hand_only.py --use-piper \
  --hand-config config/hand.yaml \
  --hand-port /dev/serial/by-id/TRIMORPH
```

Its raw episode contains `camera.mp4`, `camera_timestamps.csv`,
`leader_hand.csv`, `piper.csv` (or `quest.csv`), `mocap.csv`, optional
`follower.csv`, and `metadata.json`. Convert raw episodes with (add
`--calibration-json` for Quest or mocap poses):

```bash
python3 data_collection/postprocess_dataset.py data_collection_0
```

See `data_collection/PIPELINE.md` for the end-to-end collection and
postprocessing workflow, `data_collection/DATASET_SPEC.md` for the processed
format, and `data_collection/README.md` for additional implementation details.

## Verification

```bash
python3 -m compileall data_collection scripts tests
python3 -m unittest discover -s tests
```

These checks do not exercise hardware. Before a full run, perform a low-current
open/mid/fist smoke test and compare the leader target, follower command, and
follower measurement for every motor.

## Learned policy inference

See [inference/README.md](inference/README.md) for the GPU predictor, Trimorph/Piper
executor, coordinate conventions, hardware-free link test, and timing limitations.
The executor uses the sibling `trimorph_hand_api` checkout in leader coordinates.
