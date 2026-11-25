# HG-DAgger with a pedal takeover for the TRIMI flow/shortcut policy

## Summary

The policy runs on the robot. When it starts to go wrong, the operator steps on the takeover pedal and finishes the episode: a human-gated takeover, as in HG-DAgger (Kelly et al. 2019). Each successful rollout with a takeover is exported whole, with every frame labeled `policy`, `human` or `handoff`. The next policy trains from scratch on D0 and the exported rollouts.

**Decisions:**
- **Both takeovers**, chosen per run with `--takeover`:
  - `teleop`: the Quest moves the arm, the worn leader TriMI moves the follower hand.
  - `kinesthetic`: the arm runs the calibrated gravity compensation and the hand goes limp, so the operator moves both directly.
- **One-way.** After a takeover the operator finishes the episode; control never returns to the policy.
- **Three pedals:** takeover, success, failure. Holding the failure pedal discards the episode.
- **Training data (2026-11-24):** the whole rollout, with no preintv label and no source weights; train from scratch. Every frame trains except a kinesthetic takeover's ~1 s handoff (takeover pedal to `active`: the arm held by the releasing hold, the hand limp), which is labeled and kept out of training windows; `--train-handoff` keeps it too. `--trim` remains for correction-only (strict HG-DAgger) exports.

**Status:** implemented and unit-tested without hardware; nothing has run on the robot yet. Before the first round, go through the bench checklist below, especially the kinesthetic entry and exit.

## Why the training objective doesn't change

HG-DAgger changes only the data; `compute_loss` (`unet_image_shortcut_policy.py:141-162`) is untouched. Three properties of the flow/shortcut policy decide how the data is mixed:
- **Multimodal fit.** A correction that differs from D0 in a similar state adds a mode. It is not averaged with the old one, as MSE behavior cloning would do.
- **The failing mode stays.** Corrections add modes; they don't remove the failing one. Policy frames up to the takeover also train, and without source weights the corrections count in proportion to their frames.
- **Chunking.** A window's target is the next 16 rows (about 1.07 s at 15 Hz). The trainer skips every window that touches a handoff frame, so a window that starts where the policy was failing never gets "stop and let the fingers go" as its target. The last ~1 s of policy frames before a kinesthetic takeover therefore start no window; they still train as targets of earlier windows. Teleop has no handoff: windows run straight from the policy into the correction.

## Running a round

### 0. Pedals

```bash
python -m scripts.piper_api.foot_pedal          # press each pedal: prints its key code
```

The executor's default `--pedal-keys a b c` is takeover, success, failure. If yours sends other keys, override them with `--pedal-keys`.

### 1. Predictor (GPU machine)

Unchanged command. The predictor now also sends the checkpoint's path, sha256 and inference steps in every reply header. Each episode stores them, so rounds can't get mixed up.

### 2. Executor (robot machine), from `trimorph-piper-control/`

DAgger-specific defaults allow this command with explicit executor settings:

```bash
bash inference/executor/run_kinesthetic.sh --predictor GPU_HOST \
  --rtc --rtc-max-guidance 5 --start-home home_multi \
  --hand-config config/xl330_new_hand_transparent.yaml \
  --hand-port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBEO3G5-if00-port0 \
  --dagger data_dagger_task_r1
```

`--dagger` selects kinesthetic takeover, pedals `a b c`, and the existing
underpowered/forced friction allowances. RTC, guidance, start and hand settings
are independent and explicitly supplied above. `--show` is independent. All settings can be overridden;
strict checking uses `--no-allow-underpowered --no-gravity-force`. Round numbers
can live in the folder name; `--dagger-round` is optional.
Keep `--gravity-comp-repo` for an explicit override, or use automatic discovery:
`TRIMORPH_GRAVITY_COMP_REPO`, sibling checkout, then the known home paths.

```bash
# Teleop takeover: operator wears the leader TriMI with the Quest controller on it.
python inference/executor/executor.py --predictor GPU_HOST --rtc --rtc-max-guidance 5 \
  --start-home home_cube_newhw --hand-config FOLLOWER.yaml --hand-port FOLLOWER_PORT \
  --dagger data_dagger_<task>_r1 --dagger-round 1 --takeover teleop \
  --teleop-calibration CAL.json --leader-port LEADER_PORT --leader-config LEADER.yaml

# Kinesthetic takeover: gravity comp from midas-piper-control (norman/piper-midas-gravity-comp).
python inference/executor/executor.py --predictor GPU_HOST --rtc --rtc-max-guidance 5 \
  --start-home home_cube_newhw --hand-config FOLLOWER.yaml --hand-port FOLLOWER_PORT \
  --dagger data_dagger_<task>_r1 --dagger-round 1 --takeover kinesthetic \
  --allow-underpowered --gravity-force
```

**Async or RTC.** Prefer `--rtc` or `--async-inference`:
- both read the pedal every action, and an RTC takeover drains the predictor's pending reply only after the operator has control;
- the synchronous loop reads it between actions, never while waiting for the predictor.

**Kinesthetic flags.**
- `--allow-underpowered` and `--gravity-force` are what drag_mode needs for the 2026-11-22 calibration: J1 and J2 are capped at 7.2 N·m, and the friction section was saved forced.
- `--piper-firmware` must match the calibration's driver (`v183`); the executor checks.

**Leader.** Use the same leader TriMI and YAML that D0 was collected with. The executor refuses a leader/follower pair whose signs or homing tables differ (`trimi_teleop.check_pair`).

### 3. Controls

| State | Takeover pedal | Success pedal | Failure pedal |
|---|---|---|---|
| At the start pose | start a rollout | | |
| Policy | **take over** | end: success | end: failure (held 2 s: discard) |
| Teleop takeover | held: clutch engaged, the arm follows the Quest; released: the arm holds | end: success | end: failure (held 2 s: discard) |
| Kinesthetic takeover | | end: success | end: failure (held 2 s: discard) |

**After the end pedal:**
1. The rollout is saved.
2. The arm and hand return to the start (hand torque off during the arm return, as the pedal reset does today).
3. The executor waits for the takeover pedal to start the next rollout.

**Other ways a rollout ends:**
- `--episode-max-s` ends a policy-only rollout as a failure.
- Ctrl-C saves the rollout in progress as `interrupted`.

### 4. Postprocess and export

```bash
python3 scripts/dagger/export_dataset.py data_dagger_<task>_r1 --out data_dagger_<task>_r1_human
# Correction-only alternative:
# python3 scripts/dagger/export_dataset.py data_dagger_<task>_r1 --out data_dagger_<task>_r1_trimmed --trim
# DESTRUCTIVE alternative, with collection/training stopped:
# python3 scripts/dagger/export_dataset.py data_dagger_<task>_r1 --overwrite-raw
```

The exporter selects complete, successful raw rollouts with human intervals, copies
each selected rollout to staging, calls the shared `postprocess_dataset.py` pipeline
with `--pose-source piper_fk` (FK of the joints on their CAN receive clock, as
kinesthetic collection), then exports either full rollouts or trimmed corrections:
- **Default (no `--trim`):** one episode per qualifying rollout, with the unchanged full video, full hand/arm arrays and syncs covering EVERY video frame. The syncs label each frame in `control`:
  - `policy`: before the takeover pedal was read;
  - `human`: from the takeover to the end (teleop: including the hand ramp and clutch-released pauses; kinesthetic: from `active`, including the freeze after the end pedal);
  - `handoff`: kinesthetic only, from the takeover to `active`: the arm is held by the releasing position hold and the hand is limp, so nobody drives the robot.

  Every frame trains except the handoff: `train_mask` (False on handoff frames) makes the trainer skip every window that touches one, and `--train-handoff` keeps them too. `intervention` marks the human frames. Teleop rollouts label the hand with the follower's measured angles until the hand ramp after the takeover ends (the worn leader was idle before it), then with the leader's; the original leader stream stays in `leader.csv`. `segments.json` counts each rollout's frames per label.
- **`--trim`:** each human interval becomes a separate compact episode. Only correction frames remain, after the start trim (the teleop hand ramp, 0 kinesthetic). Videos are re-encoded, arrays sliced, and indices remapped. This mode trains on human corrections only.
- Rollouts recorded before the Piper log had CAN receive times fall back to `--pose-source piper` (poll-time FK), with a warning.
- A correction interval must have at least 32 camera rows by default (`--min-rows`). In full mode, at least one interval must meet that threshold to retain the rollout. In trimmed mode, short intervals are skipped individually.
- In both modes, camera/sample timing differences above `--max-sync-error-ms` (default 10) produce warnings and retain nearest samples, matching shared postprocessing. Invalid indices, non-increasing timestamps, and video/frame-count mismatches still fail validation.
- It always requires success AND a human correction; failed, interrupted and policy-only rollouts are excluded before processing.
- `segments.json` lists what was kept and why the rest was skipped.
- `--out` is required unless `--overwrite-raw` is given; a nonempty destination is refused and no source files are changed. Output includes raw episodes, processed arrays, videos and syncs. Raw and processed videos are hard-linked to avoid two physical copies.
- `--overwrite-raw` replaces raw/processed episodes in the original root and deletes rejected recordings and stale training caches after all exports validate. It preserves full retained rollouts unless combined with `--trim`. Other root files are preserved. Rename failures are rolled back; this is not power-loss atomic across directories. Empty exports or validation failures leave the original dataset intact. Temporary space for final outputs plus one full selected rollout is needed.
- `export_segments.py` remains an alternative entrypoint. The old `--video`, `--outcomes` and `--force` options are removed; `--out` never deletes existing output.

### 5. Train (diffusion-policies)

From scratch, every training window drawn uniformly (no source weights):

```bash
python shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_shortcut_policy \
  'tasks.dataset.dataset_path=[/data/D0,/data/data_dagger_<task>_r1_human]' \
  exp_name=trimi_dagger_r1
```

Windows that touch a handoff frame (syncs `train_mask`) are skipped in training,
validation and the normalizer fit; D0 has no mask, so all its windows train.

**Several roots.**
- Roots load in order, one replay buffer.
- Validation episodes are drawn inside each root; D0 keeps the same split as when it is loaded alone.
- Validation also logs `val_loss/<root name>`.
- Outputs go under the first root.

**Not used in this plan, still available:** `tasks.dataset.source_weights` (each root's share of the training draws) and the warm start `training.init_ckpt` / `training.keep_ckpt_normalizer`.

**Round 1 without the new options:** copy the exported episodes into the D0 root after its indices, and retrain from scratch.

### 6. Rounds

1. π0 is the current checkpoint. Round k runs π(k-1) for 20–30 rollouts from the evaluation start distribution (`--start-home`, `--start-jitter-*`).
2. **When to take over:** as soon as the behavior looks wrong. Deep failures reach states that are hard to recover from and that a better policy won't visit. Correct with the same strategy as the D0 demos.
3. **Metrics:** each episode's metadata has the outcome, takeover time and human time. Progress is the takeover rate per rollout; stop adding rounds when it stops falling.

## What the takeovers do

### Teleop (`--takeover teleop`)

- **Anchor.** The arm is anchored at the pose the policy last commanded (FK of the last joint target), not the measured one. The arm is still finishing that target, so the measured pose would mean a step back. The first teleop target equals the last policy target.
- **Mapping:** quest_teleop.py's.
  - `TeleopMapper` with orientation; clutch deltas, re-anchored at each engagement.
  - Rate limits: 300 mm/s, 180 deg/s.
  - Clamp: 300 mm and 90° around the takeover pose.
  - Disengage on a stale Quest (0.15 s), a Quest jump (50 mm) or a persistent tracking error.
- **Arm commands.** One IK step per tick at 60 Hz through the executor's `JointController`, with the same joint margins, |J5| ≥ 10° and speed caps as the policy.
- **Hand.** The leader TriMI is read at 100 Hz with torque off. The follower ramps from the policy's last hand target to the leader over 0.4 s, then follows it through the existing `HandCommandWriter`. No force feedback yet; see open items.
- **Labels.** Arm: FK of the measured joints on their CAN receive clock. Hand: the follower's measured angles until the hand ramp after the takeover ends, then the leader's angles, the operator's intent including squeeze, as in D0 (set at export).

### Kinesthetic (`--takeover kinesthetic`)

The executor runs gravity_comp's `DragController` itself, on its own CAN connection. drag_mode can't run alongside, because both would write to can0.

**Before any hardware moves:**
- The calibration artifact, gravity model and drag config are built from the same calibration `run_gravity_comp.sh` uses.
- The executor warns if gravity compensation can't hold the start pose:

  | Start | J1 torque needed (cap 7.2 N·m) |
  |---|---|
  | home_4c (the executor default) | 9.8: sags |
  | home_cube_newhw | 5.0 |
  | home_unscrew1 | 5.8 |
  | quest_0 | 2.6 |

**Supported DAgger entry (different from standalone drag_mode's gravity ramp):**
1. The operator's hands are already on the arm. Hands first, then the pedal.
2. The hand goes limp; `--kinesthetic-hand hold` keeps its last target instead, for an arm-only correction.
3. Before switching modes, every joint/motor feedback clock must advance. A missing or stalled source rejects takeover while the arm is still in position control.
4. The arm switches to MIT mode and immediately receives full calibrated gravity feed-forward plus a position hold at the measured takeover joints. Commands stream continuously; there is no application-level 0.3 s sleep before the first command. Firmware acceptance across the transition still needs a supported bench check.
5. The position stiffness releases over `--kinesthetic-ramp-in-s` (default 1 s) while gravity support remains full. Friction assistance stays off during this handoff. Begin demonstrating only after the console says `Kinesthetic takeover active`; that is when the human training interval opens.
6. Friction assist defaults to 0 (`--kinesthetic-assist-frac`) until drag_mode has validated one.

**Exit:**
1. The end pedal freezes the arm: brake with gravity retained, then a position hold with a kp ramp.
2. The arm goes back to position control at the measured joints, in the order gravity_comp's `goto_pose` uses: enable, position mode, `move_js` to the measured joints.
3. The normal return to start follows.

**Faults.**
- A runaway trip by the controller ends the episode as a failure.
- Stale feedback freezes the arm, puts it in position hold, and stops the executor. Freshness uses the joint-stream and all six motor timestamps, not unchanged position/velocity values. The error names the source (e.g. `motor5`).
- Ctrl-C freezes the arm, puts it in position hold, and stops the executor. Ctrl-C is caught during the loop; it never leaves the arm in MIT mode without a hold.

**Labels.** Arm: FK of the measured joints on their CAN receive clock. Hand: the follower's measured angles. These record no squeeze.

## Recording

Each rollout is `DATASET_ROOT/raw/episode_N/`, in the collection layout. `postprocess_dataset.py --pose-source piper_fk` reads it as is.

| File | Contents |
|---|---|
| `camera.mp4`, `camera_timestamps.csv` | every frame with its DepthAI capture time, as collection records it |
| `leader_hand.csv` | the hand label stream: leader (teleop) or follower (kinesthetic), `pos_0..pos_12` |
| `follower.csv` | follower measured angles and commanded target, 50 Hz |
| `piper.csv` | as kinesthetic collection records it: one row per joint feedback frame (polled at 250 Hz, deduplicated), with its CAN receive time on the host clock (`joint_rx_monotonic_ns`), joints, the firmware flange pose in `x..yaw`, velocities, currents, status |
| `commands.csv` | every arm target, tagged `policy` or `teleop` |
| `events.csv` | executor and DAgger events: start, takeover, engage/disengage, mit_switch, active, freeze, end |
| `chunks.npz` | the policy's chunks, as the executor run log |
| `quest.csv` / `gravity_comp.csv` | Quest packets (teleop) / every MIT cycle, source timestamps, stale source/read errors, and commanded kp/kd (kinesthetic) |
| `metadata.json` | `status`, the episode clock (`started/stopped_monotonic_ns`), counts, hand YAML sha256s, and the `dagger` block below |

The `dagger` block records:
- round, takeover kind, outcome and reason;
- the policy provenance from the predictor;
- executor mode and start record;
- segments (policy, human), `takeover_ns` and `human_intervals` (teleop clutch engaged, kinesthetic `active`); the export derives the frame labels from these;
- the takeover settings, including calibration paths and sha256s.

The Piper log and run log are emptied after each rollout, so a long session doesn't grow in memory.

## Bench checklist before the first round

**Teleop:**
- With the arm in a clear area, run the executor with `--takeover teleop`. Take over during a slow policy motion and check:
  - no jump;
  - the first teleop target is the last policy target (`commands.csv`);
  - the clutch holds the arm when released.
- Check the leader→follower ramp and the hand label stream (`leader_hand.csv`).

**Kinesthetic** (at a held start, e.g. home_cube_newhw, with the arm supported):
- **Entry:** take over while the policy is still and the arm is physically supported. Verify full gravity from the first command and smoothly decreasing position stiffness during `HANDOFF` (`gravity_comp.csv`, `piper.csv`). Software tests cannot establish when the firmware accepts the first MIT command. Do not progress to moving takeovers if it drops, jolts or buzzes.
- **Exit:** end the episode and check that the switch from the MIT hold back to position mode doesn't move the arm. This is the step most in need of a hardware check: the firmware might keep an old position target.
- **Mid-motion takeover:** repeat during policy motion; check that the runaway watchdog doesn't trip on the arm's momentum.
- **Load:** check the loop overrun warning (the camera, Piper log and hand reader run alongside).
- Check the operator's hands in the wrist camera during takeovers.

**Pipeline:** export a pilot round and load it in the trainer. Check each rollout's `control_rows` in `segments.json`, and on a few videos that the label boundaries fall where control actually changed.

## Known limits and open items

- **Visual gap.** D0's images come from the hand-held camera, DAgger's from the robot's wrist camera. Without source weights, the DAgger share of training is its share of the frames.
- **Hand state.** Training hand deltas are leader minus leader; deployment adds them to the follower's measured angles, so grips come out weaker under contact. This predates DAgger. Both hands are now recorded, which makes it measurable; if `hand_pose_abs` is re-enabled, DAgger episodes should use `follower.csv` as the observation.
- **Teleop force feedback.** None in v1. It needs `ContactDetector` / `WallContact` / `LeaderWall` (`trimi_teleop.py:263-518`) driven from the takeover loop.
- **Kinesthetic supported entry.** DAgger now streams full gravity plus a releasing kp hold from the first MIT command. This removes the software's uncommanded wait/zero-gravity ramp, but requires a supported hardware bench test; it does not establish firmware mode-latch behavior or overcome insufficient torque capacity.
- **Kinesthetic torque.** Firmware 1.8-8 or newer (16 N·m) would hold the arm at every start; the calibration would then have to be redone.
- **Ending by keyboard.** Not needed with three pedals.

## Files

| Repository | File | Change |
|---|---|---|
| trimorph-piper-control | `inference/executor/dagger.py` | new: pedals, rollout control, recording, teleop and kinesthetic takeovers |
| | `inference/executor/executor.py` | `--dagger` options; `pause_at_start` / `wait_for_policy_start` hand over to the session; policy provenance in the run log; RTC drains after a takeover |
| | `inference/predictor/predictor.py`, `inference/protocol.py` | checkpoint provenance in every reply header |
| | `scripts/piper_api/foot_pedal.py` | per-key hold tracking, key names, a key-code monitor |
| | `scripts/dagger/export_segments.py` | new: rollouts → training episodes with policy/human/handoff frame labels |
| | `tests/test_dagger.py` | new: 20 tests (real IK for teleop; real `DragController` and the real calibration for kinesthetic) |
| diffusion-policies | `shared/env/midas/midas_image_dataset.py` | several roots, per-source validation, `source_weights`, `source` in batches, `train_mask` window filtering |
| | `shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py` | weighted sampler, `init_ckpt`, `keep_ckpt_normalizer`, normalizer coverage, per-source validation loss |
| | `shortcut_policy/config/...` | the new keys; output dirs under the first root |
| | `tests/test_dagger_dataset.py` | new: 13 tests |
