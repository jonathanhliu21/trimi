# Trimorph policy deployment

This ports the MIDAS predictor/protocol/executor into this repository. The GPU
machine runs the shared diffusion-policy predictor. The robot machine runs the
Trimorph/Piper executor and the same OAK camera backend/settings as collection.
It does not import the MIDAS controller repository.

## Install

On the robot machine, in its control environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ../trimorph_hand_api
python -c 'import midas_hand_api; print(midas_hand_api.__file__)'
```

The Trimorph fork retains the `midas-hand` distribution name, `midas_hand_api`
module, and `MidasHand` class. The executor defaults to the sibling
`../trimorph_hand_api` source directory and checks the imported module location.
Use `--hand-api-repo` if it lives elsewhere. Keep the GPU training environment
separate from the hardware dependencies.

On the GPU machine, install the dependencies for your `diffusion-policies`
checkout. Run from this repository's root (the checkout can live on each machine
at a different path):

```bash
python inference/predictor/predictor.py \
  --policy-repo /path/to/diffusion-policies \
  --checkpoint /path/to/trimi/checkpoints/latest.ckpt \
  --device cuda:0
```

No stale machine-specific checkpoint is selected automatically. The predictor
loads the checkpoint's normalizer and sampling settings. To experiment with
fewer denoising passes, pass `--num-inference-steps 8` and evaluate policy quality.

## Test the connection without hardware

With the predictor running:

```bash
python inference/executor/executor.py \
  --predictor GPU_HOST --fake --max-cycles 20
```

Fake mode uses synthetic images/state and does not connect to the hand, Piper,
or camera. The real GPU model still runs. Unit tests need no GPU or devices:

```bash
python -m unittest discover -s tests -p 'test_*inference.py'
python -m unittest discover -s tests -p 'test_rtc.py'
```

## Execute on Trimorph and Piper

```bash
python inference/executor/executor.py --predictor GPU_HOST --hz 15
```

The hand defaults are the bench TriMi: `../trimorph_hand_api`,
`config/xl330_test_config.yaml`, and the FTDI `FTBEO9P0` serial port. That
adapter is shared with the glove, so confirm TriMi is on it, or pass
`--hand-api-repo`, `--hand-config`, and `--hand-port`.

Press Enter to move to the start pose, then press the foot pedal to start the
policy. Camera preview stays live while waiting. With `--no-pedal`, execution
starts as soon as setup finishes.

The hand stays torque-off during initial setup and the arm's start move. Once
the arm reaches its target, the executor writes the hand-start target before
enabling hand torque. With `--no-hand-start`, it instead restores torque at the
hand's measured pose after the arm arrives. `--no-start-move` holds the measured
hand pose immediately because no arm start move is performed.

This enables and commands the actual hand and arm. Before the first observation
the arm ramps in joint space (10 deg/s, after an Enter prompt) to a named start
from `PIPER_START_JOINTS_DEG` in `data_collection/collection/config.py`:
`--start-home home_4c` by default (logged start #1, a tested offset of `home_4b`),
`home_4d` (tested start #6), `home_4b` (median start of episodes 95-220 of
`data_trimorph_hand_mocap_quest_4`), `home_4a` (its episodes 0-94),
`home_cube1` (median start of `data_trimorph_hand_mocap_quest_5`), `home_unscrew1`
(median start of `data_trimorph_unscrew_cap_0`), or `quest_0`.
Every start reached (startup and pedal resets) is appended to
`output/executor_starts.jsonl` with an id, the commanded and measured joints, and
the measured flange pose; `--start-id N` repeats start N exactly (joints and hand
start). `--start-jitter-mm` / `--start-jitter-deg` (default 0) add a random flange
offset around the chosen start, solved by IK with replay's joint checks and logged
like any other start.
Use `--start-joints-deg J1..J6` for another start, `--start-speed-deg-s` to change
the speed (max 30), or `--no-start-move` to start wherever the arm and hand are.
After the arm, the hand moves (at the configured hand profile velocity) to the
same start's median demo hand pose, `HAND_START_POSITIONS_RAD`; a hand that is not
within 0.05 rad after 5 s only prints a warning. `--no-hand-start` moves only the arm. `--show`
opens a live wrist-camera window (flipped like the policy input); it freezes
while waiting for the predictor in synchronous mode, and q/Esc stops the executor.
Add `--record-video` to save `wrist.mp4` in the run folder under
`output/executor_runs/` (or `--log-root`). Recording starts after the initial
robot start move, uses the same flipped view as the policy at `--cam-fps`, and
continues through inference waits and pedal pauses without needing `--show`.
The video is finalized on shutdown, including Ctrl-C. It also works with
`--no-log`, which saves only the video; `--fake` does not support recording.
The arm is driven in joint space by default (`--arm-control joint`,
`inference/executor/joint_control.py`): each policy pose is solved by local IK
from the previous joint target and sent with `move_j`. The IK is exact except
near the singularity (Jacobian smallest singular value below
`--ik-damping-sigma`, default 0.03; 0.087 at `home_cube1`), where it accepts a
few mm of pose error instead of large J2/J3 jumps; each step is scaled down to
the joint speed caps (`--max-joint-speed-deg-s`, default 45 45 90 60 45 60), and
joints stay 3 deg inside the SDK limits with |J5| >= 10 deg. `--arm-control
pose` restores the firmware Cartesian `move_p`, which oscillated in J2/J3 near
the edge of reach. Joint targets are not sent as one jump per action: each is
reached over the action period by evenly spaced, linearly interpolated `move_j`
commands at about `--command-hz` (default 60: 4 per action at `--hz 15`), which
removes the per-action stop-and-go of a 15 Hz stream at the cost of about half
an action period (~33 ms) of extra lag. `--command-hz 15` sends one per action.
The predictor starts every chunk exactly at the arm pose in the observation
state (checked on run `20261104_113450`: chunk start moves 1.00x with the
observed pose). With the measured pose, the arm's tracking lag or swing at
observation time is therefore copied into each chunk. By default
(`--state-anchor command`) the observation carries the last commanded pose
instead (in joint mode, FK of the joint target sent), so each chunk continues
the commanded path; if the arm is more than `--anchor-max-mm` (50) or
`--anchor-max-deg` (20) away from it, the measured pose is sent and a warning
printed. `--state-anchor measured` restores the previous behavior. The hand
state is always measured.
Each run is logged to `output/executor_runs/<timestamp>/` (`--log-root`, or
`--no-log`): `piper.csv` polls the Piper at 200 Hz (joints, motor velocity and
current, firmware flange pose, `motion_status`), `commands.csv` has every arm
target sent (raw policy and EMA-filtered, XYZ + Euler), `chunks.npz` the raw
predictor chunks with the observed state, and `events.csv` the observe/chunk
timing. Everything is held in memory and written when the executor exits,
then `plot_run.py` saves `joints.png` and `eef.png` (position, velocity, and
acceleration of each joint and flange axis, measured vs commanded, with
predictor waits shaded), `eef_position_error.png` (signed XYZ and total 3D
position error in mm, measured FK minus requested target, sampled at target
arrival; one action period after issue for interpolated joint commands), and
`spectrum.png` into the same folder, and prints
per-band velocity RMS: measured well above commanded means the arm adds the
shake; equal means it is in the commands. Replot or zoom with
`python inference/executor/plot_run.py RUN_DIR --start 10 --end 16`.

Executor logs also include these diagnostics in the same run folder:

- `hand_commands.csv`: per executed action, the original absolute policy hand
  target, the chunk's measured hand reference, scaled target, EMA target, and
  target clipped to the connected hand's configured limits. `clipped_motor_ids`
  identifies affected motors. Cycle/index link to `chunks.npz` and `commands.csv`.
- `hand_writes.csv`: start/end timestamps of each successful hand API write and
  its clipped calibrated target. This records host writes, not acknowledgement
  that the motors reached their targets. Repeat writes and delays are visible.
- `observations.csv`: existing measured hand reads, measured flange XYZ/Euler,
  the arm reference sent to the predictor, read timing, camera frame receive
  timestamp/age, and anchor selection/fallback. It covers sync, async, and RTC.
- `wrist_timestamps.csv`, with `--record-video`: zero-based video frame indices
  and host monotonic receive timestamps to align video with the other logs.
  These are host receive times, not camera exposure timestamps. The sidecar is
  finalized when recording closes, including when `--no-log` saves video only.
- `meta.json`: both EMA values, hand action gain, start preset values, hand
  calibration/limits, source hashes, and the predictor's checkpoint provenance.
  Updated predictors also report their actual observation schema and whether
  the neural network conditions on hand positions. Restart the predictor to
  receive these extra provenance fields; older predictors still work.

Default logging reuses existing hand observation reads and adds no serial reads.
Per-motor hand gains are configured in `HAND_JOINT_ACTION_SCALES` in
`executor.py`: `None` (or an omitted motor) uses `HAND_ACTION_SCALE`, while a
number replaces the global gain for that motor. For example, `7: 0.8` applies
a gain of 0.8 to middle motor 7 even if the global gain is 1.15. Scaling applies
to displacement from the chunk's measured hand reference before EMA; zero
removes that motor's predicted displacement and targets the chunk reference.
The override map and effective gains are saved in `meta.json`.

For denser hand position, velocity, and current samples, add
`--hand-telemetry-hz 10` (allowed range 0–50; default 0). This creates rows in
`hand_telemetry.csv` while hand targets are active. Sampling uses the same serial
lock as writes and observations; it can reduce the achieved hand write rate.
Read start/end times are host transaction times, and values retain the hand
API's caching behavior. Exceptions are logged with blank unavailable readings;
diagnostic read exceptions do not stop playback. All motor positions are
calibrated radians, velocities rad/s, and currents mA in motor order 0–12.
CSV rows remain in memory until shutdown, like the existing arm logs.

`--traj-view [DATASET]` opens a 3D window with every demonstrated flange
trajectory of `DATASET` (default `data_trimorph_hand_mocap_quest_4`), the live
EEF pose and trail, the chunk being executed, and the distance to the nearest
demo point; `python inference/executor/trajectory_view.py DATASET` shows the
trajectories alone. Hand setup follows the repository's
replay path: mode 3 position control, hold measured position before enabling
torque, configured velocity/acceleration profiles, then nonblocking position
writes. It does not start the collection force-feedback workers. On exit it
shuts down the hand and disconnects Piper; it does not retract/home the arm.

Piper firmware/interface/channel/installation/payload and OAK image size, FPS,
flip, manual exposure/ISO, and white-balance defaults come from
`data_collection/collection/config.py`. Override them through `--help` options
when collection used different settings. The hand API uses the YAML baudrate,
calibration, gains, and limits. It verifies the connected motor models.

## HG-DAgger rollouts (`--dagger`)

For the current kinesthetic bench setup:

```bash
bash inference/executor/run_kinesthetic.sh --predictor GPU_HOST \
  --rtc --rtc-max-guidance 5 --start-home home_multi \
  --hand-config config/xl330_new_hand_transparent.yaml \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO3G5-if00-port0 \
  --dagger DATASET_ROOT
```

`--dagger` defaults to kinesthetic takeover, pedals `a b c` (start/takeover,
success, failure), and the current calibration's `allow-underpowered`/`gravity-force`
allowances. It does not change RTC, guidance, start pose, hand configuration,
hand port or `--show`; specify those independently as above.
Explicit options override defaults; `--no-allow-underpowered` and
`--no-gravity-force` restore strict calibration checks. Bench defaults live in
`data_collection/collection/config.py`. `--dagger-round` remains optional for
older commands; naming the dataset folder by round is sufficient.

`--gravity-comp-repo PATH` is still supported. Without it, discovery checks
`TRIMORPH_GRAVITY_COMP_REPO`, a sibling `midas-piper-control` checkout, then
`~/midas_hand/midas-piper-control` and `~/workspace/midas-piper-control`. An
explicit path or environment setting is authoritative (an invalid setting fails
instead of silently selecting another checkout). For a custom location, set:

```bash
export TRIMORPH_GRAVITY_COMP_REPO=/path/to/midas-piper-control
```

`--dagger DATASET_ROOT --takeover teleop|kinesthetic` records every rollout as a
raw episode in the collection layout and lets the operator take over:

- Three pedals (`--pedal-keys`, default `a b c`; find the codes with
  `python -m scripts.piper_api.foot_pedal`):
  - takeover: starts a rollout at the start pose, and takes over during the policy;
  - success and failure: end the rollout (hold failure 2 s to discard it).
- After a takeover the operator finishes the episode:
  - **teleop:** the Quest moves the arm, from the pose the policy last commanded, with the
    takeover pedal held as the clutch; the worn leader TriMI moves the follower hand.
  - **kinesthetic:** the arm runs the calibrated gravity compensation from
    midas-piper-control and the hand goes limp. Put your hands on the arm before you press
    the pedal.
- Use `--rtc` or `--async-inference` so the pedal is read every action.

Export directly from raw recordings with
`python scripts/dagger/export_dataset.py DATASET_ROOT --out OUTPUT_ROOT`.
It selects successful rollouts with human corrections, invokes shared Piper
postprocessing (`--pose-source piper_fk`: FK of the joints on their CAN receive
clock, as kinesthetic collection) and validates sync alignment. By default, videos
and syncs include the full retained rollout, including autonomous policy execution,
and label every frame `policy`, `human` or `handoff`. Every frame trains except a
kinesthetic takeover's ~1 s handoff, whose windows the trainer skips (syncs
`train_mask`; `--train-handoff` keeps them). Add `--trim`
to export only human corrections with compact videos/arrays and remapped syncs.
Timing differences above `--max-sync-error-ms` (default 10) warn and retain
nearest samples, matching shared postprocessing. Invalid indices, non-increasing
timestamps, and video/frame-count mismatches still stop the export.
The source is preserved and nonempty outputs are refused. Alternatively,
`--overwrite-raw` destructively replaces the original dataset with selected
rollouts and deletes rejected episodes after validation; add `--trim` to keep
corrections only. Stop collection
and training before using that option. One of these two modes is required.
See `DAGGER_PLAN.md` at the repository root for the full procedure, the
training options, and the bench checklist.

## Coordinate contract

The executor sends the latest hand target at `--hand-command-hz` (default 100 Hz),
independently of the policy action rate (`--hz`, default 15 Hz) and Piper command
rate. It repeats each target until the next action arrives, including during
inference waits. Hand reads and writes share a lock on the serial bus. The writer
pauses for start-pose resets and stops before hand shutdown. During a pedal reset,
hand torque is disabled while the arm returns to start. The executor writes the
hand start target before enabling hand torque again, then waits for the hand to
reach it. With `--no-hand-start`, it restores torque at the measured hand pose
after the arm returns. If the arm return fails, hand torque remains off.
This rate is a host
scheduling target; serial latency and reads can delay writes.

- State and absolute actions are `[XYZ_m, rotvec_rad, hand_rad]`, shape `[19]`.
- Hand elements are exactly motor IDs `0..12`, in leader coordinates. Use the
  same signs, home offsets, and zero calibration as collection. There is no
  follower scaling, thumb retargeting, or passive-joint/16-DOF expansion.
- `read_pos()` supplies calibrated hand positions. `set_positions()` applies
  hardware signs/home offsets once and clips to configured joint limits.
- Piper feedback Euler xyz angles are converted to rotation vectors before
  inference; predicted rotation vectors are converted back to Euler xyz for
  Piper commands. They must never be passed through as interchangeable angles.
- Camera frames are flipped like collection and converted from OAK BGR to RGB.
- Quest-derived training poses should be processed with the actual
  `--piper-start-pose` so orientation-dependent frame conversion agrees with
  execution. The inference protocol does not carry pose-frame metadata.
- `--hz` must match the synchronized training sample rate after downsampling.
  A 30 FPS dataset with stride 2 uses 15 Hz, not the raw hand measurement rate.

## Timing and limitations

Before playback, the executor checks the entire action chunk for Euclidean XYZ
jumps greater than `MAX_EEF_JUMP_M = 0.15` meters, defined in
`inference/executor/executor.py`. It checks the observation-to-first-action jump
and every consecutive action pair. Any violation prints a warning and stops
the executor without applying any hand or arm actions from that chunk. This
check does not enforce workspace, table clearance, joint limits, or rotation limits.

By default the executor uses synchronous request/reply execution. It captures an
observation, waits for the predictor, and then executes the returned chunk.
Inference therefore remains a pause between chunks by default. Enable overlapping
inference on the executor with:

```bash
python inference/executor/executor.py --predictor GPU_HOST --hz 15 \
  --async-inference --inference-latency-ms 70
```

The predictor command and protocol stay the same. To test the link without
hardware, add `--fake --max-cycles 20 --no-log` to this command.

Use `--execute-steps K` to execute at most the first K actions of each returned
chunk, in either synchronous or async mode. The default `0` uses the entire
returned chunk. If fewer than K actions arrive, use the available count. Raw
chunks are still logged and validated in full, while the trajectory view shows
the actions that will actually execute. In async mode, skipped entries count
toward this prefix: `--execute-steps 8` with a start index of 2 executes indices
2 through 7, not eight additional actions.

In async mode, the executor forecasts the current chunk's finish from its
remaining **chosen** actions and `--hz`, and captures the next observation
`--inference-latency-ms` before that finish (default **70 ms**). JPEG encoding,
network I/O, and prediction run in the background while playback continues.
For N chosen actions at frequency f, nominal execution takes `N / f` seconds,
so the next request starts about `max(0, N / f - latency_seconds)` after playback
starts. The deadline is recalculated after every action from its actual send
time and the number of chosen actions remaining. For example, with
`--hz 15 --execute-steps 8`, an available 8-action prefix takes about 533 ms and
the next request starts about 463 ms into playback with the 70 ms lead.
`--command-hz` controls interpolation within each action; it does not multiply
the number of policy actions or change this duration calculation.
The estimate should cover the full request/reply latency, including inference;
increase it if replies routinely arrive late. It is a fixed tunable estimate,
not an automatically measured delay. A value of 0 requests at the chunk end.

When a reply is ready, it replaces the previous chunk **at the next policy action
boundary**: the current action's interpolated arm commands finish, then all
unexecuted old actions are discarded. The first action of the new chunk plays
next after skipping its elapsed entries, with EMA history retained. The start
index is `ceil(observation_age_seconds / action_dt)`, where `action_dt = 1 / hz`
and returned index 0 represents the observation. Age is measured on a monotonic
clock from the start of the state read to takeover, including state-read,
encoding, network, inference, and action-boundary waiting time. It does not use
the fixed `--inference-latency-ms` request lead or the predictor's `infer_ms`.
For example, a 10-action chunk at 100 Hz lasts
100 ms: the next request starts around 30 ms with a 70 ms estimate. If the reply
arrives at 35 ms, it takes over at 40 ms and drops the six remaining old actions.
The new prediction is then 10 ms old, so playback begins at its index 1.
The first prediction after startup or a pedal reset keeps index 0 because it
has no moving predecessor; startup latency does not consume its horizon.

If the start index reaches or exceeds the returned length or selected prefix,
the entire replacement is discarded as stale and a fresh observation is
requested. The executor continues the old tail if available, otherwise holds
the last target. It never clamps to a stale final action. Such discarded
requests still count toward `--max-cycles`.

There is at most one request in flight. A chunk shorter than the lead time starts
its next request after issuing its first action. The first chunk requires an
initial wait; if a later reply is late, the chosen prefix finishes and the robot
holds its final target until the reply arrives, without playing the excluded
suffix. Playback never speeds up to catch
up. Hardware reads, writes, and GUI updates stay on the control thread; async
camera observations use the newest available frame without blocking for a new
one. A pedal reset discards the old tail and any in-flight reply, drains that
reply, and takes a fresh observation after resuming. `--max-cycles` counts
requests, including any reply discarded on reset.

The predictor still anchors absolute actions to the state captured **at request
time**. Skipping aligns the new chunk in time but does not guarantee that
independent predictions choose the same trajectory. There is no rebasing or
continuation guidance in plain async mode. The whole-chunk jump check also
checks the last commanded action against the selected takeover action, rather
than against the discarded index 0.

Each cycle logs observation time, full predictor `infer` time, request/reply
`rtt`, execution time, and action deadline overruns. `rtt` includes prediction;
do not add the two values. `observe + rtt` is the approximate inter-chunk overhead
on top of the scheduled action interval in synchronous mode. Predictor timing covers CPU output
conversion as well as model execution. Socket and camera timeouts bound waits;
invalid/nonfinite outputs or mismatched sequence numbers abort execution.

Async cycle logs include planned, executed, skipped, and dropped action counts;
dropped includes both skipped entries and an unexecuted tail. `chunk_start`
records observation age and the selected index; `chunk_discarded` records stale
predictions. Command indices retain their original positions in the raw chunk.
In async
run plots, gray spans mark actual playback waits rather than inference that
overlaps motion. Synchronous and RTC logging retain their existing behavior.

The `--async-inference` and `--rtc` flags are mutually exclusive. With neither
flag, execution remains synchronous. `--execute-steps` applies only to synchronous
and plain async execution; RTC uses `--rtc-horizon` for replanning timing.

### Real-time chunking (`--rtc`)

`--rtc` removes the pause with real-time chunking (Black, Galliker, Levine
2025, "Real-Time Execution of Action Chunking Flow Policies"). Start the
predictor from this repository; it answers both kinds of requests:

```bash
python inference/executor/executor.py --predictor GPU_HOST --rtc
```

The predictor returns the whole prediction horizon (16 actions), and the executor
runs each chunk from index 1, one action per period. Once `--rtc-horizon`
(default 8) of its actions have run, it sends the next observation right after
commanding action k0, without stopping. The arm reference matches plain async
and synchronous execution: `RobotIO.read_state()` uses the last filtered/
reachable commanded pose, after EMA, IK and joint-speed limiting. In joint mode
this is FK of the action's joint endpoint; in Cartesian mode it is the filtered
Cartesian command. The arm pose is used for relative-to-absolute action
conversion, not as a neural-network observation input. `--rtc` still requires
`--state-anchor command`.

The request carries the previous absolute targets from k0 on plus the expected
delay d, the most actions run while waiting for any of the last 10 replies.
The predictor expresses those targets relative to the new reference using the
existing SE(3) conversion. It does not shift the old absolute trajectory to a
new location or filter the overlap a second time. A difference between the
filtered reference and the raw plan does not discard overlap guidance.

The fallback check also matches plain async: if measured pose is beyond
`--anchor-max-mm` / `--anchor-max-deg` from that commanded pose,
the measured pose is sent with no overlap,
so the policy replans from the arm instead of continuing the plan. The arm keeps executing the old chunk. The reply takes over at the
first period after it arrives, at index m + 1 after m more old actions. If the
old chunk runs out first, the arm holds its last target ("starved" in the log).
Only the first chunk after the start or a pedal reset is waited for.

`state` events report `command; measured off ...` during normal tracking, and
`measured; FALLBACK ...` when the translation or orientation tracking threshold
is exceeded. Reference selection is changed only in the RTC request path;
plain async and synchronous behavior are unchanged. No predictor logic update
is required. This reference correction does not establish that all release
overtravel is fixed; compare with `--rtc-max-guidance 0` to isolate the reference
from overlap guidance, which still favors the old absolute trajectory.

The predictor samples the new chunk with inpainting guidance toward the old
one (`inference/predictor/rtc.py`): weight 1 on the observation and the d
actions run before it arrives, exponentially decaying weights over the rest of
the overlap, 0 after, applied with pseudoinverse guidance (PiGDM) at each sampling
step. Flow/shortcut policies guide the endpoint estimate `x_t + (1-t) v_t`
inside the flow integrator, preserving its time and shortcut-distance inputs.
Diffusion policies retain the DDIM/DDPM guidance path. Unsupported policy
samplers fail explicitly. Each guided step back-propagates through the UNet,
so expect a longer `infer`. `--rtc-max-guidance` clips the guidance and `0`
disables it. The default remains 20 for existing diffusion deployments; start
with `--rtc-max-guidance 5` when evaluating flow/shortcut policies, especially
with few sampling steps. PiGDM is approximate, so the new chunk follows the old one
closely but not exactly. Each takeover prints the delay (and the expected one),
the step from the last old action to the first new one next to the old chunk's
own step at that point ("old plan"), and how far the new chunk is from the old
actions run while it was predicted; `events.csv` keeps the same numbers as
`rtc_takeover`. In run plots the gray spans then mark inference in flight; the
arm keeps moving during them.

The executor checks the selected takeover action against the last commanded
action and stops if the XYZ jump exceeds `MAX_EEF_JUMP_M`, even when all adjacent
steps inside the new prediction pass validation. This check does not impose
hand-joint or angular continuity limits.

Deploy `inference/predictor/rtc.py` on the predictor machine together with the
updated `shortcut_policy/shortcut_model.py` and
`shortcut_policy/unet_shortcut/unet_image_shortcut_policy.py` from
`diffusion-policies`. Restart the predictor; existing checkpoints need no
retraining. Deploy `inference/executor/executor.py` on the robot machine for
the takeover check. Use `--rtc` without `--async-inference`;
`--inference-latency-ms` is only used in plain async mode.

RTC still captures observations and performs socket I/O on the control thread;
it can hold or miss motion deadlines under camera/network delays. The plain
async mode uses a socket worker, but does not apply RTC guidance.

No hardware performance or policy success is established by unit tests or the
synthetic link test.
