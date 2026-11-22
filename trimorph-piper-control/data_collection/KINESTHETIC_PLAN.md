# Plan: kinesthetic collection with gravity comp, then a replay identical to the rollout

## Context

Data collection is moving from mocap and Quest to kinesthetic teaching on the robot itself:
- The TRIMI hand is mounted on the Piper.
- The arm runs the gravity compensation calibrated on 2026-11-22 (midas-piper-control `gravity_comp.drag_mode`, calibration `trimorph-piper-control/scripts/grav_comp/calib/robot1_trimi.json`).
- The hand's torque is off, so the operator moves the fingers.

Each episode is then replayed on the robot to check its quality. That replay has to command the arm and hand exactly as the policy rollout does.

Why what exists today isn't enough:
- **Connect frames.** `collect_hand_only.py --use-piper --no-mocap` was built for the firmware drag-teach button. When it connects, the Piper worker sends a 0x151 mode frame with the MIT flag off and a 0x477 payload frame (`collection/piper_worker.py:66-73` → `scripts/piper_api/piper_api.py:196-234`). While drag mode holds the arm in MIT torque mode, either frame could drop or jerk the arm. `--piper-joint-level` adds 0x472 queries, and `--piper-start-*` commands motion.
- **Pose source.** It logs poses from the firmware end-pose stream, which nobody has checked still flows in MIT mode. Joints are optional, and torques and arm status aren't logged at all.
- **Replay vs rollout.** The existing replays (`data_collection/replay_episode.py`, `replay_actions.py`) differ from the rollout in many ways:
  - IK: `replay_episode.py` defaults to firmware Cartesian `move_p` at 250 Hz. Its `--arm-control joint` mode uses the rollout IK at 15 Hz with 60 Hz joint writes, but still differs in everything below.
  - Sampling: continuous interpolation instead of the trainer's stride-2 sync rows.
  - No EMA, chunking or re-anchoring.
  - Hand: targets scaled ×1.1, and interpolated at 60 Hz instead of a 15 Hz hold resent at 100 Hz.

Outcome:
- One launcher for collecting with gravity comp.
- Datasets whose arm poses come from the same kinematic model the rollout IK inverts.
- A replay that runs each episode through the deployed executor itself, so every setting is the rollout's by construction.

## Decisions

- **Two processes, one launcher:** drag mode commands the arm, and the collector only listens.
- **Start pose:** move it to one the arm holds by itself, and use it for both collection and rollout.

## Part 1: kinesthetic collection with gravity comp (first)

**Architecture: two processes.** Drag mode runs unchanged, with its own terminal UI, safety logic and SDK build, and it is the only process that writes to can0. The collector only listens. SocketCAN gives every socket a copy of every frame, and the arm broadcasts its feedback without being polled, so a listener can't disturb drag mode.

### 1.1 Receive-only arm mode in the collector

Files: `data_collection/collect_hand_only.py`, `collection/piper_worker.py`, `collection/episode.py`, `collection/config.py`

- **New flag.** `--arm-mode gravity-comp` implies `--use-piper --no-mocap`.
- **Sends nothing.** It forces installation and payload to `unchanged` (no 0x151 or 0x477), skips the 0x472 limit queries, rejects `--piper-start-*`, and never enables or disables the arm. These are cached reads only (`PiperAPI.get_joint_state` / `get_motor_state` / `get_state`).
- **Logs one row per CAN feedback sample,** deduplicated on the joint frames' receive time (about 200 Hz; today it polls at 250 Hz and about 20% of rows are duplicates). Each row has:
  - joints
  - joint velocity and current-derived torque
  - arm status, control mode and error code
  - drive enable bits
  - the firmware end pose when it is streaming.
- **Metadata.** `piper.mode = "gravity_comp_kinesthetic"`, calibration path and sha256, the launcher's drag-session id, and the hand YAML's offsets, signs and limits (not only its hash).
- **Monitor box.** An arm box in the tracking monitor stays green while all drives are enabled with no error code. Episodes where that changes are flagged.
- **Tests.**
  - A mocked `PiperAPI` asserts zero transmitting calls in this mode (`tests/test_piper_worker.py`).
  - Flag combinations (`tests/test_hand_only_collection.py`).

### 1.2 Hand: torque off, same calibration as the rollout

Files: `collection/leader_hand_worker.py`, `collection/hand_common.py`

- Use the same hand and YAML as the rollout logs (`config/xl330_new_hand_transparent.yaml` on `/dev/ttyUSB0`). Read through `MidasHand.read_pos()`, which is the exact inverse of the rollout's `set_positions` (`trimorph_hand_api/midas_hand_api/hand.py:189-200, 385-388`).
- While torque is off, put the motors in operating mode 3 (the executor's mode), so position reads are comparable with the rollout's.
- Count, per episode and motor, the samples outside the YAML limits. The rollout clips those; for example motor 8 is limited to ±0.01, and motor 11's upper limit is 0.

### 1.3 Bench check on the arm, before building further

- With drag mode running and the arm in the held region, start the collector in the new mode. Confirm:
  - drag mode shows no freeze or overruns;
  - joints stream at about 200 Hz.
- Record whether the firmware end pose streams in MIT mode.

### 1.4 Launcher

New file: `scripts/kinesthetic/run_kinesthetic_collection.sh`. Reuse `scripts/grav_comp/run_gravity_comp.sh`'s CAN check and its midas repo and venv discovery.

- **Preflight.**
  - CAN is up.
  - The hand is mounted as it was for stage 4a (adapter, controller, camera, cables dressed).
  - The gravity_comp unit tests pass.
- **Calibration copy.** Copy the calibration into the session folder and run drag mode on the copy. Drag mode's runtime write then doesn't modify the versioned file, and each dataset keeps the calibration it used.
- **tmux pane 1:** `gravity_comp.drag_mode --force --allow-underpowered`. The operator does the ritual there.
- **tmux pane 2:** the collector with `--arm-mode gravity-comp`, `--hand-config`, `--hand-port` and `--dataset-root`. Start it once drag mode prints "assist at target".
- **Stop order (printed):** stop the collector so the episode is saved. Then press `q` in drag mode, support the arm, and press Enter.

### 1.5 Start pose the arm holds by itself (decided)

Files: `collection/config.py`, `collection/tracking_monitor.py`, plus a small check script in `scripts/grav_comp/`

- **New named start.** Add one to `PIPER_START_JOINTS_DEG` and `HAND_START_POSITIONS_RAD`, chosen around home_cube_newhw or home_unscrew1, where gravity comp holds the arm. Use it for both collection and rollout. Rollout selects it with `--start-home` in the kinesthetic profile; the global default (`home_4c`) stays, because other datasets use it.
- **Torque check.** The check script computes, from the calibration's gravity model, the J1–J6 torque that pose needs and fails if J1 or J2 exceeds about 6.5 N·m (a margin under the 7.2 cap). The launcher runs it in preflight.
- **Guide.** The collector can't move the arm in this mode. The monitor shows the per-joint error to the start (green within tolerance), and the measured start joints are recorded in metadata.

### 1.6 Poses from joints

File: `data_collection/postprocess_dataset.py`

- New `--pose-source piper_fk`. `absolute_action_t` is the FK of the logged joints from `piper_model()` (`data_collection/replay_actions.py:246`, the pyAgxArm MDH model). That is the model `inference/executor/joint_control.py:JointController` inverts, so replaying a pose through the rollout IK returns the recorded joints.
- Sync uses the CAN receive time. Joints stay in `processed/episode_N/piper.csv`.
- When the firmware pose is present, report how far it is from the FK pose.

### 1.7 Episode checks

File: `scripts/dataset/check_episodes.py` (today it never reads `piper.csv`)

- Piper rate, gaps and sync distance.
- Arm-status changes.
- Joints near their limits.
- Hand samples outside the YAML limits.
- Time with J1/J2 torque at the 7.2 N·m cap, meaning the operator was carrying the arm.

Docs: add a kinesthetic gravity-comp section to `data_collection/PIPELINE.md`.

## Part 2: replay identical to the rollout (second)

**Approach.** Replay each episode as an "oracle policy" through the real executor (`inference/executor/executor.py`). A dataset predictor answers the executor's requests with the episode's ground-truth chunk, so everything downstream is the deployment code:
- anchoring and EMA (arm 0.5, hand 0.9)
- `JointController` IK at 15 Hz, 4 linear substeps (60 Hz joint writes), speed caps
- `HandCommandWriter` at 100 Hz, YAML clipping, hand mode and profile
- start move and the sync/async/RTC chunking modes.

Only the action source differs.

### 2.1 Dataset predictor

New file: `inference/predictor/dataset_predictor.py`

- **Protocol.** Same wire protocol and port (`inference/protocol.py` `recv_obs` / `send_action`).
- **Sampling** follows the trainer's `MidasImageDataset` (`diffusion-policies/shared/env/midas/midas_image_dataset.py:443-481`):
  - rows `sync[first_cut:][::stride]`
  - pose `absolute_action_t[pose_idx]`, hand `processed_encoder_t[encoder_idx]`
  - horizon 16, with the last row repeated past the end (pad_after 7).
  - Defaults come from the trimi task config: stride 2 (15 Hz), first_cut 0.
- **Chunk.** At row pointer p, take the window p..p+15, convert it with `se3_utils.absolute_to_relative`, then with `relative_to_absolute(rel, state)` using the request's state. This is the same conversion as `diffusion-policies/shared/env/midas/midas_inference.py:125`. Reply with 8 actions to plain requests and 16 to RTC requests, as the real predictor does.
- **Pointer.** It advances by the chunk index that was executing when the observation was taken: 7 per chunk in sync mode, `k0` in RTC (`executor.py:1319-1324`).
- **Latency.** `--reply-latency-ms` emulates the deployment's observation-to-chunk delay, so the pauses at chunk boundaries match. The default comes from the last rollout's `events.csv`, about 72 ms.
- **End.** After the last row it holds the final pose and reports the episode finished.

### 2.2 Executor additions, with the control path unchanged

Files: `inference/executor/executor.py`, `inference/executor/async_predictor.py`, `inference/protocol.py`

- The obs header carries `chunk_index` at the sync (`executor.py:929`), RTC (`executor.py:1241, 1322`) and async send sites. `predictor.py` ignores it.
- `--hand-start-rad` (13 values) next to `--start-joints-deg`, so the existing start move can use the episode's first hand sample.
- `meta.json` additionally logs the hand EMA alpha, `HAND_ACTION_SCALE`, the hand YAML sha256, the git commit and the predictor kind (policy or dataset).

### 2.3 Simulated backend for offline replays

New file: `inference/executor/sim_backend.py`, plus a `--sim` flag in `executor.py`

- Inject a fake Piper and a fake hand into the real `RobotIO`, so IK, substeps, EMA and the hand writer run unchanged without hardware. The fakes track perfectly:
  - measured joints = the last write
  - pose = `piper_model()` FK
  - hand reads = the clipped command.
- The existing `--fake` doesn't do this, because its `FakeRobot` skips IK (`executor.py:766-788`).

### 2.4 Replay launcher

New file: `data_collection/replay_as_rollout.py`

- **Inputs:** dataset and episode, plus `--like-run output/executor_runs/<ts>`, which copies every executor flag from a real rollout's `meta.json`.
- **Default profile** is the last deployment: sync mode, `--hz 15`, `--command-hz 60`, `--hand-command-hz 100`, joint control, Piper speed 100, hand 3 rad/s and 50 rad/s², transparent YAML.
- **Start:** by default, the episode's first logged joints and hand sample, reached through the executor's normal start move. `--start-home NAME` uses the rollout's named start instead.
- **Before starting:**
  - Refuse while a drag-mode process is running.
  - Check the hand YAML hash against the episode's metadata.
- **Runs:** the dataset predictor on localhost, then the executor with `--predictor 127.0.0.1`. Add `--sim` for an offline run.

### 2.5 Replay report

New file: `scripts/dataset/replay_report.py`

- **Inputs:** the executor's run folder (`piper.csv`, `commands.csv`, `events.csv`, `chunks.npz`) and the episode.
- **Per episode it reports:**
  - replayed vs demo arm joints, aligned by action index
  - IK residual (requested pose vs the FK of the IK target)
  - speed-cap saturation
  - anchor fallbacks and jump stops
  - hand commanded vs measured vs demo
  - clipped hand targets
  - a pass/flag line.
- Plots reuse `inference/executor/plot_run.py`.

`replay_episode.py` stays as the raw replay tool, including its `--arm-control joint` local-IK mode.

## Order of work

1. 1.1 and 1.2, with unit tests.
2. 1.3 bench check on the arm.
3. 1.4 to 1.7.
4. Pilot collection of a few episodes.
5. 2.1 to 2.3, with unit tests.
6. Offline replay of the pilot episodes with `--sim`.
7. 2.4 and 2.5, then replay on the robot.

## Verification

- **Unit tests:**
  - `python3 -m unittest discover -s tests` in trimorph-piper-control.
  - The gravity_comp tests run through the launcher's preflight.
- **Cross-repo test** in `diffusion-policies/tests/`, following `test_trimi_piper_pipeline.py`: the dataset predictor's windows equal `MidasImageDataset`'s for the same episode.
- **Pointer and protocol tests:**
  - The pointer logic in sync, async and RTC modes, driven by the executor's loops with a fake robot.
  - `predictor.py` ignores the new header field.
- **Bench and pilot:**
  - The drag-mode log shows no freezes during collection.
  - `check_episodes.py` passes.
  - The difference between the FK and firmware poses is small.
- **Offline:** `replay_as_rollout.py --sim` on a pilot episode. Commanded joints should track the demo joints, except where the report flags caps or IK.
- **Robot:** replay one pilot episode with `--like-run` pointing at the last rollout. The replay's `meta.json` and that rollout's should differ only in predictor kind and start pose.

## Risks

- **Torque cap.** Gravity comp only holds the arm where J1/J2 need at most 7.2 N·m.
  - The new start pose avoids the problem at the start of each episode.
  - Demos that reach into the folded-back region (J2 above about 120°) or swing J1 past about −60° still sag there, and the operator has to carry the arm. 1.7 flags those episodes.
  - Firmware S-V1.8-8 or newer (16 N·m) would remove the cap, but the calibration would have to be redone.
- **Hands in the camera.** The operator's hands appear in the wrist camera during demos, which the rollout never sees. Keep them out of view where possible.
- **Grasps.** Finger positions recorded at contact give little squeeze under position control. The replay shows whether grasps hold.
- **End pose in MIT mode.** Whether the firmware end pose streams in MIT mode is unverified (1.3). The FK poses from joints don't depend on it.
