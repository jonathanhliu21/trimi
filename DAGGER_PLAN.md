# HG-DAgger with a pedal takeover for the TRIMI flow/shortcut policy

## Summary

The policy runs on the robot. When it starts to go wrong, the operator steps on the takeover pedal and finishes the episode. Only the human-controlled part becomes training data (HG-DAgger, Kelly et al. 2019). The next policy trains on D0 ∪ H1 ∪ … ∪ Hk.

**Decisions:**
- **Both takeovers**, chosen per run with `--takeover`:
  - `teleop`: the Quest moves the arm, the worn leader TriMI moves the follower hand.
  - `kinesthetic`: the arm runs the calibrated gravity compensation and the hand goes limp, so the operator moves both directly.
- **One-way.** After a takeover the operator finishes the episode; control never returns to the policy.
- **Three pedals:** takeover, success, failure. Holding the failure pedal discards the episode.
- **Training data:** only human-controlled intervals. Policy rows are recorded for metrics.

**Status:** implemented and unit-tested without hardware; nothing has run on the robot yet. Before the first round, go through the bench checklist below, especially the kinesthetic entry and exit.

## Why the training objective doesn't change

HG-DAgger changes only the data; `compute_loss` (`unet_image_shortcut_policy.py:141-162`) is untouched. Three properties of the flow/shortcut policy decide how the data is mixed:
- **Multimodal fit.** A correction that differs from D0 in a similar state adds a mode. It is not averaged with the old one, as MSE behavior cloning would do.
- **The failing mode stays.** It is not removed, only made less likely as the corrections gain weight. That is why each batch draws a fixed share from the DAgger roots.
- **Chunking.** A window's target is the next 16 rows (about 1.07 s at 15 Hz). Each human interval is exported as its own episode, so no target reaches into rows the policy controlled.

## Running a round

### 0. Pedals

```bash
python -m scripts.piper_api.foot_pedal          # press each pedal: prints its key code
```

The executor's default `--pedal-keys b c a` is takeover, success, failure: the middle, right and left pedal of a PCsensor 3-pedal as shipped. If yours sends other keys, pass them, for example `--pedal-keys 48 46 30`.

### 1. Predictor (GPU machine)

Unchanged command. The predictor now also sends the checkpoint's path, sha256 and inference steps in every reply header. Each episode stores them, so rounds can't get mixed up.

### 2. Executor (robot machine), from `trimorph-piper-control/`

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
python3 data_collection/postprocess_dataset.py data_dagger_<task>_r1 --pose-source piper
python3 scripts/dagger/export_segments.py data_dagger_<task>_r1        # -> data_dagger_<task>_r1_human
```

The exporter turns each human interval into one training episode:
- It keeps the sync rows inside the interval, after a trim at its start: 400 ms for teleop (operator reaction and hand ramp); 0 for kinesthetic, whose intervals already start after the gravity ramp-in.
- It skips intervals under 32 camera rows.
- It keeps only rollouts that ended as a success; change that with `--outcomes success failure`.
- `segments.json` lists what was kept and why the rest was skipped.

### 5. Train (diffusion-policies)

```bash
python shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py \
  --config-name=train_unet_image_trimi_hand_cond_dinov2_shortcut_policy \
  'tasks.dataset.dataset_path=[/data/D0,/data/data_dagger_<task>_r1_human]' \
  'tasks.dataset.source_weights=[0.8,0.2]' \
  training.init_ckpt=/path/to/previous/checkpoints/latest.ckpt \
  training.keep_ckpt_normalizer=true \
  training.num_epochs=100 optimizer.lr=5e-5 training.lr_warmup_steps=100 \
  exp_name=trimi_dagger_r1
```

**Several roots.**
- Roots load in order, one replay buffer.
- Validation episodes are drawn inside each root; D0 keeps the same split as when it is loaded alone.
- Validation also logs `val_loss/<root name>`.
- Outputs go under the first root.

**Weights.**
- `source_weights` sets each root's share of the training draws (a `WeightedRandomSampler`); weights can also be given by name.
- Pick the share from the per-root `train windows` the trainer prints, so that a DAgger window is drawn about 10–20 times as often as a D0 window:

  share ≈ r · n_dagger / (n_D0 + r · n_dagger), with r = 10–20.

  Example: 150 demos of about 20 s are about 45,000 windows; 10 takeovers of about 5 s are about 700. That gives a share of about 0.2.
- Raise the share as rounds accumulate, up to about 0.5. Plain aggregation (r = 1) drowns the corrections. A much larger share repeats the same few windows thousands of times and forgets the nominal behavior.
- Keep the batch size of the D0 training (256); the share, not the batch size, sets the mix.

**Warm start.**
- `init_ckpt` loads only the model and EMA weights.
- `keep_ckpt_normalizer` keeps the checkpoint's normalizer, which the predictor un-normalizes with. The trainer prints the share of normalized DAgger targets outside [-1, 1]; above 1%, refit the normalizer and train from scratch instead.

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
- **Labels.** Arm: the measured flange pose (FK of the measured joints). Hand: the leader's angles, the operator's intent including squeeze, as in D0.

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

**Entry, as in drag_mode:**
1. The operator's hands are already on the arm. Hands first, then the pedal.
2. The hand goes limp; `--kinesthetic-hand hold` keeps its last target instead, for an arm-only correction.
3. The arm switches to MIT mode. The firmware holds no torque for the 0.3 s mode latch.
4. Gravity ramps in over `--kinesthetic-ramp-in-s` (default 1 s, the minimum `DragConfig` allows).
5. Friction assist defaults to 0 (`--kinesthetic-assist-frac`) until drag_mode has validated one.

**Exit:**
1. The end pedal freezes the arm: brake with gravity retained, then a position hold with a kp ramp.
2. The arm goes back to position control at the measured joints, in the order gravity_comp's `goto_pose` uses: enable, position mode, `move_js` to the measured joints.
3. The normal return to start follows.

**Faults.**
- A runaway trip by the controller ends the episode as a failure.
- Stale feedback freezes the arm, puts it in position hold, and stops the executor.
- Ctrl-C freezes the arm, puts it in position hold, and stops the executor. Ctrl-C is caught during the loop; it never leaves the arm in MIT mode without a hold.

**Labels.** Arm: FK of the measured joints. Hand: the follower's measured angles. These record no squeeze.

## Recording

Each rollout is `DATASET_ROOT/raw/episode_N/`, in the collection layout. `postprocess_dataset.py --pose-source piper` reads it as is.

| File | Contents |
|---|---|
| `camera.mp4`, `camera_timestamps.csv` | every frame with its DepthAI capture time, as collection records it |
| `leader_hand.csv` | the hand label stream: leader (teleop) or follower (kinesthetic), `pos_0..pos_12` |
| `follower.csv` | follower measured angles and commanded target, 50 Hz |
| `piper.csv` | 200 Hz: `x..yaw` = FK of the measured joints, plus joints, firmware pose, velocities, currents, status |
| `commands.csv` | every arm target, tagged `policy` or `teleop` |
| `events.csv` | executor and DAgger events: start, takeover, engage/disengage, mit_switch, active, freeze, end |
| `chunks.npz` | the policy's chunks, as the executor run log |
| `quest.csv` / `gravity_comp.csv` | Quest packets (teleop) / every MIT cycle (kinesthetic) |
| `metadata.json` | `status`, counts, hand YAML sha256s, and the `dagger` block below |

The `dagger` block records:
- round, takeover kind, outcome and reason;
- the policy provenance from the predictor;
- executor mode and start record;
- segments (policy, human), with `human_intervals`, the training intervals;
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
- **Entry:** take over while the policy is still. Measure the sag across the 0.3 s mode latch and the 1 s ramp-in (`gravity_comp.csv`, `piper.csv`).
- **Exit:** end the episode and check that the switch from the MIT hold back to position mode doesn't move the arm. This is the step most in need of a hardware check: the firmware might keep an old position target.
- **Mid-motion takeover:** repeat during policy motion; check that the runaway watchdog doesn't trip on the arm's momentum.
- **Load:** check the loop overrun warning (the camera, Piper log and hand reader run alongside).
- Check the operator's hands in the wrist camera during takeovers.

**Pipeline:** postprocess and export a pilot round; load it in the trainer with the old normalizer and read the coverage line.

## Known limits and open items

- **Visual gap.** D0's images come from the hand-held camera, DAgger's from the robot's wrist camera. Weight the DAgger share with that in mind.
- **Hand state.** Training hand deltas are leader minus leader; deployment adds them to the follower's measured angles, so grips come out weaker under contact. This predates DAgger. Both hands are now recorded, which makes it measurable; if `hand_pose_abs` is re-enabled, DAgger episodes should use `follower.csv` as the observation.
- **Teleop force feedback.** None in v1. It needs `ContactDetector` / `WallContact` / `LeaderWall` (`trimi_teleop.py:263-518`) driven from the takeover loop.
- **Kinesthetic warm entry.** Full gravity from the first MIT command plus a kp handover would remove most of the entry sag. It needs a `DragController` change upstream (midas-piper-control) and a bench test.
- **Kinesthetic torque.** Firmware 1.8-8 or newer (16 N·m) would hold the arm at every start; the calibration would then have to be redone.
- **Ending by keyboard.** Not needed with three pedals.

## Files

| Repository | File | Change |
|---|---|---|
| trimorph-piper-control | `inference/executor/dagger.py` | new: pedals, rollout control, recording, teleop and kinesthetic takeovers |
| | `inference/executor/executor.py` | `--dagger` options; `pause_at_start` / `wait_for_policy_start` hand over to the session; policy provenance in the run log; RTC drains after a takeover |
| | `inference/predictor/predictor.py`, `inference/protocol.py` | checkpoint provenance in every reply header |
| | `scripts/piper_api/foot_pedal.py` | per-key hold tracking, key names, a key-code monitor |
| | `scripts/dagger/export_segments.py` | new: human intervals → training episodes |
| | `tests/test_dagger.py` | new: 14 tests (real IK for teleop; real `DragController` and the real calibration for kinesthetic) |
| diffusion-policies | `shared/env/midas/midas_image_dataset.py` | several roots, per-source validation, `source_weights`, `source` in batches |
| | `shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py` | weighted sampler, `init_ckpt`, `keep_ckpt_normalizer`, normalizer coverage, per-source validation loss |
| | `shortcut_policy/config/...` | the new keys; output dirs under the first root |
| | `tests/test_dagger_dataset.py` | new: 12 tests |
