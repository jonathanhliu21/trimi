# Data collection and postprocessing pipeline

This guide follows one dataset from recording to the files the diffusion-policy
trainer reads. Run every command from the repository root with `.venv`
activated. Details live in [README.md](README.md) (collector options, replay,
mocap) and [DATASET_SPEC.md](DATASET_SPEC.md) (the trainer's contract).

```text
  TRIMORPH glove ─┐
  Quest (UDP) ────┤   collect_hand_only.py    raw/episode_N/     check_episodes.py    postprocess_dataset.py
  Vicon (ROS 2) ──┼──▶  one thread per   ──▶  CSV + MP4 per ──▶  quality report  ──▶  camera/  processed_encoder_t/
  OAK-1 camera ───┤     stream, host          stream, saved                           absolute_action_t/  syncs/
  Piper (opt.) ───┘     monotonic clock       on pedal release                        processed/  (replay)
```

## 1. Before collecting

| Check | How |
| --- | --- |
| Glove bus at 1 Mbaud, return delay 0 | Set on all 13 motors in Dynamixel Wizard; `baudrate: 1000000` in `config/xl330_test_config.yaml`. At 57600 baud the hand records 25 Hz instead of 50 Hz. |
| USB latency timer 1 ms | `cat /sys/bus/usb-serial/devices/ttyUSB*/latency_timer` prints 1 (udev rule `99-dynamixel-ft232h-latency.rules`). |
| Glove port | `ls /dev/serial/by-id/` shows the glove adapter (FTBEO9P0). |
| Quest streaming and tracking | The Quest sender targets UDP port 5005. The tracking monitor (below) must show the Quest box green while the controller moves. |
| Vicon plate visible | The monitor's VICON box is green, not `OBJECT LOST`. The collector starts the ROS 2 relay itself. |
| Start pose (needed for absolute or mocap poses) | Add `--piper-start-home`: the Piper moves to the teleop home after an Enter prompt, and its measured flange pose is saved in every episode. |

## 2. Collecting

For direct arm demonstrations without mocap or Quest, record the arm's measured
6-DoF flange feedback alongside the hand and camera:

```bash
python3 data_collection/collect_hand_only.py \
  --dataset-root data_piper_fk --use-piper --no-mocap \
  --hand-config config/xl330_test_config.yaml --hand-port /dev/serial/by-id/YOUR_GLOVE
python3 data_collection/postprocess_dataset.py data_piper_fk --pose-source piper
```

Use the arm's built-in drag-teach button to demonstrate. The logger reads
`PiperAPI.get_eef_state()` / SDK `get_flange_pose()` (arm-reported Cartesian
feedback); it does not compute a separate FK model on the host. Raw `piper.csv`
contains base-frame XYZ in metres and roll/pitch/yaw in radians. No external
tracking, calibration, or start-pose anchoring is needed. Arm joint logging
remains off unless `--piper-joint-level` is supplied; even with it, training
actions use the six Cartesian pose fields. These poses describe the flange;
a tool-tip offset would need a separate transform.

### Kinesthetic teaching under gravity comp (receive-only arm logging)

With the TRIMI hand mounted on the Piper, `gravity_comp.drag_mode`
(midas-piper-control, calibration `scripts/grav_comp/calib/robot1_trimi.json`)
holds the arm in MIT torque mode while the operator moves it, and the hand's
torque is off. Use the launcher from the repository root:

```bash
scripts/kinesthetic/run_kinesthetic_collection.sh \
  --define-start --passive-j6 --dataset-root data_trimorph_kinesthetic
```

Each task gets its own recorded arm and hand start. `--define-start` is the
launcher's default when no profile or explicit legacy start is supplied. It
shows `DEFINE START`, with no saved target or target skeleton. After assist is
at target, position the arm and hand for `multi_pick_and_place`, hold that pose,
and use the pedal to record a short stationary clip (about 1–2 seconds). The
operator chooses the capture moment. The definition episode is marked
`start_definition` and is not a training demonstration.

Once the clip is saved, extract its measured start offline. Replace
`episode_XXXXXX` with that definition episode:

```bash
.venv/bin/python scripts/kinesthetic/capture_task_start.py \
  --task multi_pick_and_place \
  --from-episode data_trimorph_kinesthetic/raw/episode_XXXXXX \
  --output config/kinesthetic_starts/multi_pick_and_place.json
```

The default sample is at episode start; use `--at-s SECONDS` to select a later
stationary moment. The file preserves measured arm joints, hand positions and
motor IDs, the hand YAML hash, source timestamps and warnings; it does not clip
the measurements. Capture does not prove that the pose is suitable for rollout.
Review its warnings, particularly the unresolved hand-limit findings below.
The current `multi_pick_and_place` profile is saved at that path from the
6.772-second definition episode 000002. Its arm target passes the nominal
torque/limit check (J1 +5.935, J2 -2.915 N*m). Recorded hand motors 4, 7 and 8
are outside YAML limits; their raw values and warnings are preserved. The
profile can guide receive-only collection, but its hand target is not cleared
for rollout.

Use that saved profile to collect this task's demonstrations:

```bash
scripts/kinesthetic/run_kinesthetic_collection.sh --passive-j6 \
  --start-profile config/kinesthetic_starts/multi_pick_and_place.json \
  --dataset-root data_trimorph_kinesthetic
```

For a later policy rollout, after collection and drag mode have stopped and
supported disable has completed, select the same profile with the executor:

```bash
inference/executor/run_kinesthetic.sh \
  --start-profile config/kinesthetic_starts/multi_pick_and_place.json \
  --predictor PREDICTOR_HOST
```

The executor validates the recorded hand target against its configuration and
limits before motion; an unreplayable target is rejected, not silently clipped.
The kinesthetic executor wrapper requires an explicit task profile or legacy
start and selects no universal task start.

The launcher checks CAN, the hand configuration and the gravity-comp tests.
With a saved profile it also checks calibrated torque at that profile's arm
pose. Definition mode has no target yet: its start-torque check is explicitly
`not_checked`, not passed. Confirm the mounted payload when prompted. A tmux session
opens with drag mode on the left and collection on the right; the collector
waits for drag mode's "assist at target" message. Complete the support/arming
ritual in the drag pane. Switch panes with Ctrl-b then an arrow key.
The default uses the foot pedal for episodes and keeps the camera on.
`--episode-s 30` instead records one timed episode; `--no-camera` is for bench
work only. `--dry-run` prints commands; `--preflight-only` runs the checks
without starting either process. `--midas-repo` selects a different gravity-comp
checkout. Both processes use their own repository's Python environment.

Each launch stores logs, a session manifest, an initial calibration snapshot,
and the drag session's working calibration under `DATASET/sessions/SESSION_ID`.
With `--start-profile`, it also saves the exact task JSON and its SHA-256; both
the preflight check and collector use that session snapshot.
Drag mode updates that copy, leaving the versioned calibration alone. The
launcher refuses a second drag process on the same CAN interface. If drag exits
or prints a freeze/stop message, the supervisor interrupts the collector so it
saves the episode. Readiness is based on drag-mode terminal output; keep watching
that terminal. Detaching tmux leaves its processes running.

`--arm-mode gravity-comp` implies `--use-piper --no-mocap`. The collector
then sends nothing on the CAN bus: no 0x151 or 0x477 frame on connect (it
refuses `--piper-installation-pos`/`--piper-payload` values other than
`unchanged`), no 0x472 limit queries, and no start move (`--piper-start-*` is
refused). Drag mode stays the only writer. Every new joint feedback frame
(about 200 Hz) becomes one `piper.csv` row with joints, velocity, current and
torque, the drive enable bits, arm status and error code, and the firmware end
pose when it arrives. The hand defaults to the rollout's
`config/xl330_new_hand_transparent.yaml` on `/dev/ttyUSB0`, read torque-off in
the executor's operating mode 3; the YAML's mode is restored on exit. The
monitor shows drive/error health as secondary text. Episodes where the expected
drive enable state or error status is unhealthy get `"arm_not_ok"` in `metadata.json`'s
`flags`. Stop the collector first, so the episode is saved, then press `q` in
drag mode, support the arm and press Enter at the disable prompt. Do not kill
the tmux session while drag torque is live.

The monitor guides the arm to the saved task start by showing flange position
distance and total rotation error from the recorded start. Its prominent START
box and filled banner are red `NOT AT START` until the flange is within 10 cm
and 45 degrees with healthy, fresh feedback; then they turn green and show
`AT START TASK`. The position check uses Euclidean distance; orientation uses
the shortest 3D rotation angle, rather than separate Euler-axis differences.
Both poses use the executor's SDK MDH flange model in Piper base coordinates.
Feedback health is secondary text, so healthy feedback alone cannot make start readiness
green. Move the arm there by hand before pressing the pedal; recording has no
start-position interlock.
The collector never commands this start. Metadata records the task, explicit
target joints, profile snapshot/hash, and episode's first measured joints; the
quality report compares against those recorded target values and thresholds.
The hand pose is not part of this arm readiness check. The limits can be set on
the collector with `--kinesthetic-start-position-tolerance-m` (default 0.10) and
`--kinesthetic-start-orientation-tolerance-deg` (default 45). Launch preflight
requires the saved pose's J1/J2 nominal commands below 6.5 N*m and every joint
within its limits and torque cap. This checks the pose, not the approach or the
whole workspace. `--start-home NAME` (launcher) and `--kinesthetic-start NAME`
(direct collector) remain explicit legacy options; `kinesthetic` was used in
the older trials below and is no longer a default for new tasks. A direct
collector without an explicit target has no start guide. The executor wrapper
refuses to run against an active drag controller; unrelated executor defaults
remain unchanged.

Check every pilot episode, then process its poses from joints:

```bash
.venv/bin/python scripts/dataset/check_episodes.py data_trimorph_kinesthetic_0
.venv/bin/python data_collection/postprocess_dataset.py data_trimorph_kinesthetic_0 \
  --pose-source piper_fk
```

Add `--no-camera` to postprocessing only for camera-free recordings. The quality
report checks arm rate/gaps, status changes, SDK joint limits, hand YAML limits,
cross-stream synchronization, and time near the J1/J2 current-derived torque
cap. High measured torque suggests saturation/support; it does not prove either.
Warnings return exit status 1 and should be reviewed before training or replay.

`piper_fk` uses the executor's pyAgxArm MDH model and synchronizes on
`joint_rx_monotonic_ns`. Processed `piper.csv` keeps the joints and motor/status
columns, preserves receipt time as `recorded_monotonic_ns`, and preserves the
firmware poses as `firmware_x` through `firmware_yaw`; its main pose fields are
FK poses. Processing metadata reports FK/firmware differences and their timing
offsets. Firmware poses are optional for FK processing. The 2026-11-22 listener
bench passed at 199 Hz; its static recording is not a cleared training pilot
(joint-limit, torque and hand-limit warnings remain). Plan Part 2, replay through
the deployed executor, follows pilot collection.

For the free-wrist profile checked on 2026-11-22, add `--passive-j6`:

```bash
scripts/kinesthetic/run_kinesthetic_collection.sh --passive-j6 \
  --start-profile config/kinesthetic_starts/multi_pick_and_place.json \
  --dataset-root data_trimorph_kinesthetic
```

This keeps all six drives electrically enabled and retains the original MIT
command sequence. J6 receives zero requested kp, kd, velocity and torque in
drag, braking and hold. Firmware quantization means a zero request is not
exactly zero physical torque. Support the free wrist throughout, and support
the whole arm during the four-second ramp from zero feed-forward. J1/J2 are
still torque-capped, so operator support remains necessary. Collection logs
all six joints, expects all drives enabled, and guides all six to the named
start. Metadata records `gravity_comp.passive_j6: true`,
`zero_torque_joints: [6]` and `disabled_joints: []`. The matching controller
changes are carried in `scripts/kinesthetic/midas_disable_j6.patch`; preflight
checks the selected option is available.

To make both wrist joints passive, use `--passive-j5-j6` instead:

```bash
scripts/kinesthetic/run_kinesthetic_collection.sh --passive-j5-j6 \
  --start-profile config/kinesthetic_starts/multi_pick_and_place.json \
  --dataset-root data_trimorph_kinesthetic
```

All six drives remain electrically enabled. J5 and J6 receive zero requested
MIT kp, kd, velocity and feed-forward torque in every control state, including
braking, freeze-hold and exception braking. This also removes their gravity,
friction and software damping torque commands. J1-J4 commands and all six-joint
feedback/guards remain unchanged. Support both free wrist joints throughout;
freeze-hold holds J1-J4 only. The collector records `passive_j5_j6: true`,
`zero_torque_joints: [5, 6]`, and `disabled_joints: []`. This option is mutually
exclusive with `--passive-j6` and `--disable-j6`, overrides J5's `--j5-kd` at
the send boundary, and has only been verified offline. It does not resolve the
reported startup drop; review the startup incident in the kinesthetic plan
before another powered trial.

The collection launcher defaults to J5 MIT `Kd=0.005` during ramp-in and drag.
Pass `--j5-kd 0` on the launcher to request zero J5 MIT damping; J5's active
`Kp` is already zero, and gravity feed-forward and software damping still apply.
The selected value is forwarded to drag mode and saved in the session command.
Other joints retain their normal gains, and J6 stays
at zero requested gains in the passive profile. Software damping, braking
and freeze-hold gains are unchanged. Firmware v183 encodes this J5 request
as approximately 0.003663. The exact launch command and derived damping
vector are saved with the session. Standalone drag mode defaults to 0.01
unless the override is explicitly supplied. The 2026-11-22 camera pilot exercised
this override on hardware with no operator-reported behavior issues; its data
remain flagged for the separate quality findings below.

`--disable-j6` is a separate, mutually exclusive diagnostic that electrically
disables J6. Its hardware trial left J1-J5 stiff and did not pass validation;
use `--passive-j6` for the next pilot. A safety freeze is latched: Enter
cannot resume assist; press q and follow the support/disable prompt.
Automatic freezes print their reason and stop collection. Before restarting,
resolve the reported fault. In the first electrically-disabled wrist trial, J6 approached
360 degrees and joint feedback disappeared; the SDK rejects readings outside
+/-360 degrees. Position the wrist within its configured joint range before
arming; do not mask this by wrapping angles or changing the encoder zero.
The +55.604-degree J6 target belonged to
the legacy `kinesthetic` trials; new tasks use their recorded profile target.

The first camera pilot (`data_kinesthetic_pilot_20261122`) is **flagged**, with
start-pose, J1 cap, joint/hand-limit and camera-to-hand sync issues; see the
pilot findings in `KINESTHETIC_PLAN.md`. Do not treat its processed output as
cleared training data. The subsequent passive-J6 camera episode
`data_trimorph_kinesthetic/raw/episode_000000` also remains diagnostic because
of start mismatch, joint/hand-limit, J1-cap and synchronization findings.
A USB latency timer of 16 ms produced alternating 16/32 ms hand-read intervals
in those recordings. On 2026-11-22 the operator set it to 1 ms with sudo, and
readback confirmed `1`. The repeat `episode_000001` had 0/862 retained camera
frames beyond 10 ms from a hand sample (maximum 8.56 ms), improved from
106/703 and 15.74 ms in episode 000000. It still has a legacy-start mismatch,
J5 limit excursions, J1 torque-cap exposure and hand-limit violations; both
episodes remain diagnostic. Recheck the timer after a USB
reconnect; if needed, set it in an operator terminal with the arm stopped:
`echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer`.

Keep the camera enabled for the image-policy trainer. In the diffusion-policies
repository, train on this output using the existing Trimi configuration:

```bash
python diffusion_policy/unet/train_unet_image_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_policy \
  tasks.dataset.dataset_path=/absolute/path/to/data_piper_fk
```

Postprocessing writes absolute XYZ + rotation-vector arrays, and the trainer
converts them to window-relative SE(3) poses plus 13 hand-joint deltas (19
action dimensions). `--no-camera` supports processing/replay only with these
image-policy configs. `--pose-source auto` (the default) selects Piper when
no calibration is supplied, preserving the existing command behavior.

For the transparent XL330 hand on `/dev/ttyUSB0`, this complete test sequence
uses the exact folder `data_trimorph_test_kinesthetic`:

```bash
python3 data_collection/collect_hand_only.py \
  --hand-config config/xl330_new_hand_transparent.yaml \
  --hand-port /dev/ttyUSB0 \
  --dataset-root data_trimorph_test_kinesthetic \
  --use-piper --no-mocap

python3 data_collection/postprocess_dataset.py \
  data_trimorph_test_kinesthetic --pose-source piper

python3 data_collection/replay_episode.py 0 \
  --dataset-root data_trimorph_test_kinesthetic \
  --hand-config config/xl330_new_hand_transparent.yaml \
  --hand-port /dev/ttyUSB0 --playback-speed 0.5
```

Collect with the pedal, then stop the collector before postprocessing/replay.
`--dataset-root` appends episodes if the folder already exists. Replay `0`
selects `episode_000000`; change the index for another episode or omit it for
the latest. Replay enables and moves both devices, prepositions at the recorded
start, and uses processed Cartesian poses without reanchoring. Arm Euler angles
are unwrapped before interpolation to avoid jumps at ±pi. No mocap is opened
unless `--use-mocap` is explicitly requested during replay. Hand targets retain
the configured `ACTION_SCALE` / `JOINT_ACTION_SCALES` in
`collection/replay_config.py` and hand joint-limit clipping.

```bash
python3 data_collection/collect_hand_only.py \
  --hand-config config/xl330_test_config.yaml \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO9P0-if00-port0
```

By default this records the leader hand, Quest, Vicon mocap, and the OAK-1 wrist
camera. Each run writes to a new folder `<name>_<n>`; the name defaults to
`DEFAULT_DATASET_NAME` in `collection/config.py` and is printed at startup
(`Saving episodes to ...`).

| Option | Effect |
| --- | --- |
| `--dataset-name NAME` | save this run to the next free `NAME_n` |
| `--dataset-root PATH` | append episodes to an existing folder |
| `--piper-start-home` | move the Piper to the start pose and record it |
| `--mocap-topic /vicon/Yori/Yori` | record only the hand-held marker plate |
| `--no-quest`, `--no-mocap`, `--no-camera` | drop a default stream |
| `--use-piper` | record Piper drag-teach poses instead of the Quest |
| `--arm-mode gravity-comp` | receive-only arm logging while gravity_comp drag mode holds the arm (see section 2) |
| `--episode-s 30` | one fixed-length episode instead of the pedal |
| `--monitor-calibration calibration.json` | show Quest (and Vicon, if calibrated) motion in Piper axes |

Episode control with the foot pedal:

| Action | Result |
| --- | --- |
| press | recording starts after 0.2 s |
| press and release | episode saved |
| hold 2 s | episode deleted |
| `q`/Escape in a window, or one Ctrl-C | collection stops; the current episode is saved |

Press Ctrl-C only once and wait for `Saved episode ...`. The streams are held in
memory and written only when an episode is saved; a second Ctrl-C during
shutdown, or closing the terminal, loses the episode except for its video.

Two windows show the run: the camera preview and the tracking monitor. The
monitor draws the Quest and Vicon paths in 3D with one status box per stream:
green with the receive rate while tracked, red with `NO SIGNAL`,
`OBJECT LOST` (Vicon cannot see the plate), or `POSITION FROZEN` (the Quest lost
positional tracking and repeats its last position). Do not demonstrate while a
box is red. Watch the terminal for repeated `read: ... no status packet` lines:
they mean the glove bus failed and the hand values are frozen copies.

## 3. What is recorded

Each saved episode is `raw/episode_XXXXXX/`. Every row carries `monotonic_ns`,
the host's monotonic clock shared by all streams, which is what synchronization
uses.

| File | Contents | Rate |
| --- | --- | --- |
| `leader_hand.csv` | TRIMORPH calibrated motor angles `pos_0..pos_12` (rad), passive, torque off | 50 Hz |
| `camera.mp4`, `camera_timestamps.csv` | OAK-1 1280x720 video; per frame `monotonic_ns` = DepthAI mid-exposure capture time, `receipt_monotonic_ns` = host arrival (~20 ms later), `sequence_num` | 30 Hz |
| `quest.csv` | raw controller packets `seq, px, py, pz, qx, qy, qz, qw`, unconverted | ~72 Hz |
| `mocap.csv` | Vicon `Position` rows (plate pose in mm, quaternion, `frame_number`) and `MarkerPositions` rows (each marker, `occluded`) | ~220 Hz received |
| `piper.csv` | Piper flange `x..yaw` (and joints with `--piper-joint-level`) | 250 Hz, only with `--use-piper` |
| `piper.csv` (`--arm-mode gravity-comp`) | joints, `vel_*`, `current_*`, `torque_*`, `enable_*`, arm status, `err_code`, `arm_ok`, firmware `x..yaw` when streaming; `robot_timestamp_s` / `joint_rx_monotonic_ns` = CAN receive time | one row per joint frame, ~200 Hz |
| `follower.csv` | MIDAS measured angles | only with a follower |
| `metadata.json` | status, stop reason, durations, sample counts, hardware settings and config hashes, `piper.start_pose`, errors | |

## 4. Checking

```bash
python3 scripts/dataset/check_episodes.py data_collection_1_1
```

The report gives each stream's rate, largest gap, and coverage, then flags:
hand rate below target, long runs of identical hand rows (still hand or bus
dropout), dropped camera frames, a frozen or jumping Quest, Vicon lost-object
samples, and episodes without `metadata.json` (stopped before saving). It ends
with a per-episode summary and exits 1 when anything is flagged. Delete episodes
that cannot be used before postprocessing; postprocessing stops at an episode
folder without `metadata.json`.

## 5. Postprocessing

```bash
# Quest as the arm pose source
python3 data_collection/postprocess_dataset.py data_collection_1_1 --calibration-json calibration.json
# Quest absolute in the Piper base, no start pose: a calibration with orientation.controller_rotation
# (fit_quest_calibration.py saves it) from the same Quest tracking session as the data
python3 data_collection/postprocess_dataset.py DATASET --calibration-json CAL --absolute

# Vicon marker plate as the source (needs a calibration fitted with --reference mocap);
# --absolute keeps the Vicon-observed flange pose in the Piper base and needs no start pose
python3 data_collection/postprocess_dataset.py DATASET --calibration-json CAL \
  --pose-source mocap --mocap-topic /vicon/Yori/Yori --absolute
```

Add `--episode N` for one episode and `-f` to overwrite. Raw files are never
modified. For each episode with `status: complete` the postprocessor:

1. **Loads the hand** from `leader_hand.csv` as a float32 `(N, 13)` array in
   motor order 0-12, unfiltered.
2. **Converts the arm pose to the Piper base frame.**
   - Quest: `p(t) = p_start + A (q(t) - q(0))` for position and
     `R(t) = B (Q(t) Q(0)^-1) B^T R_start` for orientation, with `A` and `B` from
     the calibration. Motion is relative to the first sample, so no Quest
     origin is needed.
   - Vicon: drops lost-object samples, re-times samples from Vicon frame
     numbers (Wi-Fi delivery is bursty), removes single-frame spikes and
     smooths (Gaussian, sigma 20 ms), then maps the plate pose to the flange
     with the calibration's Vicon-to-Piper and plate-to-flange transforms.
   - Piper (no calibration): the recorded flange pose as is.

   The start pose `p_start, R_start` is `--piper-start-pose`, else the pose
   recorded with `--piper-start-home`. Without either, a Quest episode starts at
   zero position and identity rotation and is marked
   `relative_to_episode_start`.
3. **Writes the action array** `absolute_action_t` as float32 `(M, 6)`:
   XYZ in metres plus a rotation vector (the trainer's SE(3) convention).
4. **Synchronizes.** One sync row per camera frame, dropping the first and last
   frame; each row takes the hand sample and the pose sample nearest in time to
   that frame. Pairs further apart than 10 ms (`--max-sync-error-s`) are
   reported but kept; nothing is interpolated.
5. **Validates** shapes, finiteness, index ranges, and video readability, then
   writes `processing_summary.json`.

## 6. Outputs

| Path | Contents | Used by |
| --- | --- | --- |
| `camera/mp4_files/XXXXXX.mp4` | copy of the episode video | trainer |
| `processed_encoder_t/XXXXXX.npy` | hand, float32 `(N, 13)`, rad | trainer |
| `absolute_action_t/XXXXXX.npy` | arm pose, float32 `(M, 6)`, XYZ m + rotation vector | trainer |
| `syncs/XXXXXX.npz` | `frame_idx`, `encoder_idx`, `pose_idx` (trainer); `encoder_monotonic_ns`, `pose_monotonic_ns`, `start_pose` (`[x, y, z, rotvec]`, NaN when unknown), `pose_frame` | trainer, analysis |
| `processed/episode_XXXXXX/piper.csv` | full-rate pose as XYZ + roll/pitch/yaw for replay | `replay_episode.py` |
| `processed/episode_XXXXXX/metadata.json` | pose source, frame, calibration path and hash, start pose | provenance |
| `processing_summary.json` | per-episode counts and paths | |

Train with `tasks.dataset.dataset_path=<dataset folder>` in the diffusion-policy
repository; replay with `replay_episode.py N --dataset-root <dataset folder>`.

## 7. Known limitations

- **Relative Quest poses.** Without a recorded start pose, `absolute_action_t`
  is motion relative to the episode's first sample (`pose_frame =
  relative_to_episode_start`, `start_pose` NaN). The trainer only uses relative
  windows, so training is unaffected; replay anchors the motion at the arm's
  current pose. Collect with `--piper-start-home` for Piper-base poses.
- **Calibration sessions.** A Quest calibration holds only for the Quest
  tracking session it was fitted in; recentering the Quest changes its frame.
  Refit when the Quest has been reset.
- **Mocap needs its own calibration.** `--pose-source mocap` requires
  `vicon_reference` in the calibration (`fit_quest_calibration.py RUN --reference
  mocap`), and a start pose unless `--absolute`.
- **Frozen hand reads are not marked.** A failed glove read silently repeats the
  last values for all 13 motors; only the terminal log and long identical runs
  in the check report reveal it.
- **Aspect ratio.** Video is 16:9 while the trainer resizes to 320x240 (4:3),
  squeezing the image horizontally.
- **Sync tolerance is a warning only.** With a 50 Hz hand and a 72 Hz Quest the
  nearest samples are within ~9 ms; the 10 ms warning mainly flags gaps.
