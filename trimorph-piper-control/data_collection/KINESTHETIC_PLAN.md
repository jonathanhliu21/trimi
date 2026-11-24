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

**Architecture: two processes.** Drag mode runs with its own terminal UI, safety logic and SDK build, and it is the only process that writes to can0. The collector only listens. The passive-J6 option and freeze reporting require the matching controller patch described below. SocketCAN gives every socket a copy of every frame, and the arm broadcasts its feedback without being polled, so a listener can't disturb drag mode.

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
- **Monitor box.** With a saved task start, the prominent START box and filled banner turn green when flange position is within 10 cm and total rotation error is within 45 degrees, with fresh, healthy feedback. Otherwise they show red `NOT AT START`. Definition mode shows `DEFINE START` without a saved target. Drive/error health is secondary text; unhealthy episodes are flagged.
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

**Passed on 2026-11-22** with firmware `S-V1.8-5`, driver `v183`, and the camera off.
The operator confirmed assist was at target before recording and reported no
freezes, overrun warnings, or behavior changes during the recording.

```bash
.venv/bin/python data_collection/collect_hand_only.py --arm-mode gravity-comp \
  --no-camera --episode-s 30 --dataset-root /tmp/kinesthetic_bench
.venv/bin/python scripts/grav_comp/listener_bench_report.py \
  /tmp/kinesthetic_bench/raw/episode_000000
```

- 5,971 arm samples across 30.000 s: **199.0 Hz**, no duplicates, 5.0 ms median
  interval, 6.28 ms p99, 15.03 ms maximum, and no gaps over 20 ms.
- All drives enabled and healthy in every row; `MOVE_MIT(0x4)`, `NORMAL(0x0)`,
  error code 0 throughout; no episode flags.
- The finalized drag log contains 5,999 control samples during the recording,
  all `ACTIVE`, and ended with `operator quit`. Its whole-session counter
  reports 2 overruns in 48,035 cycles (0.0042%); it does not locate those events
  within the session. The operator observed no overrun warnings during the bench.
- Firmware end poses were fresh in every row. Their receive timestamps advanced
  across the full 30 s, with a maximum 4.67 ms difference from joint timestamps.
- 1,501 hand samples at 50.0 Hz over their sample span (the report shows 49.0 Hz
  because its denominator includes episode shutdown). All 13 motors responded;
  IDs 1, 6, 9, and 12 reported XC330-M181, accepted by the mixed-model check.
- Local evidence: `/tmp/kinesthetic_bench/bench_report.txt`,
  `collector_retry.log`, `raw/episode_000000/`, `calibration_used.json`
  (the snapshot matches the episode's calibration SHA256), and
  `piper-1_drag_20261123T005121Z.npz` (UTC filename).
- Use the repository `.venv`: the base Conda Python imports an older hand API.
  187 gravity-comp tests and 47 focused collection/listener/hand tests passed.
  A broader test run also found an unrelated stale camera-exposure expectation.

### 1.4 Launcher

Implemented and checked offline on 2026-11-22. The launcher uses native tmux
panes and logs, validates the exact calibration bytes it copies, and waits for
the completed "assist at target" line. A freeze/stop message or a dead drag
pane interrupts the collector to save the episode. The real tmux readiness
path was tested with harmless fake processes. Live acquisition was exercised
in the camera pilots below; their quality findings remain unresolved.

New file: `scripts/kinesthetic/run_kinesthetic_collection.sh`. Reuse `scripts/grav_comp/run_gravity_comp.sh`'s CAN check and its midas repo and venv discovery.

- **Preflight.**
  - CAN is up.
  - The hand is mounted as it was for stage 4a (adapter, controller, camera, cables dressed).
  - The gravity_comp unit tests pass.
  - A saved task profile passes the calibrated start-torque check. In definition mode, the target is unknown and this check is explicitly `not_checked`.
- **Calibration copy.** Copy the calibration into the session folder and run drag mode on the copy. Drag mode's runtime write then doesn't modify the versioned file, and each dataset keeps the calibration it used.
- **tmux pane 1:** `gravity_comp.drag_mode --force --allow-underpowered`. The operator does the ritual there.
- **tmux pane 2:** the collector with `--arm-mode gravity-comp`, `--hand-config`, `--hand-port` and `--dataset-root`. Start it once drag mode prints "assist at target".
- **Stop order (printed):** stop the collector so the episode is saved. Then press `q` in drag mode, support the arm, and press Enter.

### 1.5 Recorded start for each task

The operator requires a separately recorded arm and hand start for each task,
instead of one fixed pose for all tasks. The earlier `kinesthetic` candidate
was based on `home_cube_newhw` and nominally needs J1=5.002/J2=-3.110 N*m,
but the operator could not comfortably reach that orientation. It remains an
explicit legacy option; neither collection nor the kinesthetic executor
wrapper selects it by default. No replacement fixed pose is adopted.

The current task is `multi_pick_and_place`. Its saved definition and outstanding
hand warnings are recorded below. The implemented workflow is:

1. Launch `scripts/kinesthetic/run_kinesthetic_collection.sh --define-start --passive-j6 --dataset-root data_trimorph_kinesthetic`. Definition is also the default without an explicit start option. Complete the supported drag ritual, choose the task's arm and hand pose, hold it stationary, and record a short pedal clip (about 1–2 seconds). The operator selects the capture moment; there is no automatic live-pose capture.
2. After the clip saves, run the offline capture command below, substituting its episode directory. `--at-s SECONDS` can select a later stationary sample instead of the default episode start.
3. Collect demonstrations with `--start-profile config/kinesthetic_starts/multi_pick_and_place.json`. The launcher validates and snapshots the file and hash, checks calibrated arm torque at that target, and passes the snapshot to the collector.
4. For later policy rollout, pass the same `--start-profile` to `inference/executor/run_kinesthetic.sh` with the predictor options. The executor validates the arm and hand target before motion, including hand configuration and limits; it does not silently clip an unreplayable recorded start.

```bash
.venv/bin/python scripts/kinesthetic/capture_task_start.py \
  --task multi_pick_and_place \
  --from-episode data_trimorph_kinesthetic/raw/episode_XXXXXX \
  --output config/kinesthetic_starts/multi_pick_and_place.json
```

The profile preserves measured J1–J6 angles, all 13 hand positions and motor
IDs, the hand YAML hash, source episode/timestamps and warnings. Definition
episodes are explicitly `start_definition`, without a target or a passed
start-torque check, and are not training demonstrations. Raw capture with
warnings does not establish rollout suitability.

The saved target drives the current/start arm overlay and the prominent
`AT START TASK` / `NOT AT START` display. At the operator's request, readiness
uses end-effector pose: flange position within 10 cm of the recorded position
and total shortest rotation error within 45 degrees, with fresh, healthy feedback.
Both flange poses come from the executor's SDK MDH FK; the former per-joint
5-degree standard is no longer the readiness criterion. The hand pose is not
part of this indicator. The collector
never commands that pose or interlocks the pedal. Metadata preserves the task,
explicit target, profile snapshot/hash and measured first joints; quality
checks use those recorded targets before any legacy lookup. Direct collection
without an explicit target has no guide. Explicit `--start-home NAME` on the
launcher or `--kinesthetic-start NAME` on the collector retains legacy access.

### 1.6 Poses from joints

Implemented and exercised on the saved bench recording: CAN receive-time sync,
5,971 FK poses, preserved raw/joint columns, and firmware comparison metadata.
FK and firmware positions/orientations agree to the displayed 0.00 mm/deg
precision in that static recording. Unit tests cover FK-to-IK consistency,
missing firmware poses, invalid samples, and timestamp jitter.

File: `data_collection/postprocess_dataset.py`

- New `--pose-source piper_fk`. `absolute_action_t` is the FK of the logged joints from `piper_model()` (`data_collection/replay_actions.py:246`, the pyAgxArm MDH model). That is the model `inference/executor/joint_control.py:JointController` inverts, so replaying a pose through the rollout IK returns the recorded joints.
- Sync uses the CAN receive time. Joints stay in `processed/episode_N/piper.csv`.
- When the firmware pose is present, report how far it is from the FK pose.

### 1.7 Episode checks

Implemented and tested for clean episodes and status, timestamp, joint-limit,
hand-limit, sync and torque failures. The old listener bench is flagged for
J6 outside SDK limits, J1 near the current-derived cap, and hand motors 4/8
outside YAML limits. These do not invalidate the listener rate/health bench,
but that static episode is not a training-quality pilot.

File: `scripts/dataset/check_episodes.py` (today it never reads `piper.csv`)

- Piper rate, gaps and sync distance.
- Arm-status changes.
- Joints near their limits.
- Hand samples outside the YAML limits.
- Time with J1/J2 torque at the 7.2 N·m cap, meaning the operator was carrying the arm.

Docs: add a kinesthetic gravity-comp section to `data_collection/PIPELINE.md`.

### Pilot findings, 2026-11-22 — not passed

`data_kinesthetic_pilot_20261122/raw/episode_000000` saved a 16.1 s camera-on
episode through the launcher: 3,211 arm samples at 199.6 Hz, 805 hand samples
at 50 Hz, and 482 camera frames at 30 Hz with no drops. FK/firmware maximum
differences were 1.58 mm and 0.63 degrees. The matched drag log has 3,218
`ACTIVE` samples, but J1 was commanded at its 7.2 N*m cap throughout.

The operator reported that gravity compensation did not behave as expected.
The measured start differed from `kinesthetic` by up to 104.91 degrees, J4/J5
approached their limits, and hand motors 0/4/7/8/10/11 exceeded YAML limits.
Motor 8 (middle MCP) reached -2.80 rad while the YAML limits it to +/-0.01;
this restriction was explicitly added in commit `04275e1`, with no explanation
of its mechanical purpose. Preserve it until the hand configuration is resolved.
The USB latency timer was 16 ms, causing 77/480 retained camera frames to exceed
the 10 ms hand-sync tolerance. The operator later set it to 1 ms and readback
confirmed `1` on 2026-11-22; synchronization after the change still needs a
repeat camera pilot. Quality checks now flag start mismatch, hand-sync
failures, and incomplete hand-limit metadata explicitly.

The first startup aborted with missing SDK joint/motor feedback; the retry
read normally. Drag now checks feedback before enabling motors. The successful
session disabled normally. Evidence and control logs are kept in
`data_kinesthetic_pilot_20261122/sessions/20261123T010917Z-8c72d1cc/`.

At the operator's request, `--disable-j6` is implemented for the launcher and
the local midas `gravity_comp.drag_mode` / `hw_session` modules. It confirms
J6 disabled, enables only J1-J5, and never sends a J6 MIT frame, including
during braking or freeze-hold. All six measured joints still feed gravity/FK
and start guidance. The collector records `disabled_joints: [6]`, expects
J6 off, and flags unexpected enabling. The wrist remains free during every
stop and needs support. This option requires the matching midas changes on
other PCs; it does not resolve J1/J2 torque saturation. A clean passive-wrist
pilot is still required before Part 2.

**Passive-J6 diagnostic, 2026-11-22:** the first attempt detected unexpected
J6 enabling and stopped. The revised mode switch disables the SDK's automatic
mode-frame insertion, explicitly disables J6 after switching into MIT, and
checks fresh drive feedback before ramp-in. The transition may briefly energize
J6; the wrist needs support throughout. In retry session
`data_kinesthetic_j6_check_20261122/sessions/20261123T012341Z-e36558a2/`, live
feedback confirmed J1-J5 enabled and J6 disabled; the operator confirmed J6
was free but the other joints held rigidly. The finalized controller log
`piper-1_drag_20261123T013339Z.npz` confirms `stale feedback`, entering BRAKE
at 41.635 s and FREEZE_HOLD at 42.130 s. Its last accepted J6 reading before
the gap was 359.236 degrees with +2.926 rad/s velocity. The local SDK rejects
the entire joint-angle sample outside +/-360 degrees, making a boundary
crossing the leading explanation; rejected raw frames were not saved, so
that cause is inferred rather than directly observed. The operator stopped
and disabled normally; a receive-only check confirmed all six drives off.

Automatic stale/runaway freezes previously printed no immediate message.
The controller now prints the first freeze reason and saves a `freeze_event`
with feedback-error detail and last accepted joint angles. Enter cannot
advance assist while frozen. The launcher rejects any freeze in its session
log, including one preceding a misleading ready line, and stops collection.
The SDK's feedback validation, watchdog thresholds and torque caps are
unchanged. J6 must be physically aligned within its configured range
(gravity model +/-120 degrees; the legacy trial target was +55.604 degrees) before another
supported trial; do not wrap reported angles or reset encoder zero to hide
the mismatch. Evidence is saved with the retry session in
`diagnostic_summary.json`. The portable controller changes are in
`scripts/kinesthetic/midas_disable_j6.patch` and must be applied to the matching
midas checkout on other PCs. Validation: 197 controller tests and 94 focused
collection/integration/postprocess tests passed without hardware commands.

**Follow-up passive-J6 retry, 2026-11-22:** session
`data_kinesthetic_j6_check_20261122/sessions/20261123T013829Z-851cd4dd/`
remained `ACTIVE` for 92.410 s with zero loop overruns and ended only on
`operator quit`. The operator still reported J1-J5 stiff while enabled;
whether the joints returned to a fixed position was unclear. Do not classify
this as a controller freeze or a confirmed firmware position-hold problem.
The receive-only audit saw fresh joint feedback in all 662 samples with
J1-J5 enabled, and J6 disabled throughout that observed interval. The raw
J5/J6 CAN frames were also saved in `feedback_audit.jsonl`. All drives were
confirmed disabled after the supported stop. No pedal episode was recorded.

At the observed pose, modeled J1 gravity command was about 7.83 N*m against
the 7.2 N*m cap; the median logged J1 command was 7.2 N*m. This pose therefore
does not establish sufficient gravity-support margin. An offline test of the
installed v183 SDK, using virtual transport and intercepted sends, found
identical explicit/automatic MIT mode fields (`ctrl=1`, `move=4`, `mit=0xAD`)
and zero encoded `kp` on the five active-joint commands. That checks command
generation, not firmware application. At that point stiffness remained unresolved: check at
a supported pose below the J1/J2 caps and, if it persists, instrument the
actual transmitted mode/gain sequence before changing gains or calibration.
The finalized controller log and `diagnostic_summary.json` are saved with
this session. This retry is **not a passed pilot**.

**Further diagnosis:** `scripts/grav_comp/compare_drag_response.py` compares
that retry with the original all-six-joint run. In the original, measured
motor torque follows ramp-in; in the disabled-J6 retry it largely retains
the previous holding torque. For example, J1's mean command rises from 0.494
N*m in the first 0.5 s to 4.891 N*m at 2-3 s, while feedback stays near
6.94 N*m. During ACTIVE its command is fixed at 7.2 N*m, but measured torque
varies strongly against joint displacement. This supports a command/mode
acceptance problem below the Python control law; it does not identify the
exact firmware cause. `comparison_report.json` contains both recordings'
measurements. J1 saturation alone does not explain the observed behavior.

The upstream `--passive-j6` profile restores the original six-drive
enable and automatic MIT-mode command sequence, while centrally overriding
J6's requested torque, kp, kd and velocity to zero in drag, brake and hold.
This keeps J6 **electrically enabled**, unlike `--disable-j6`; the operator
explicitly accepted that alternative for a supported test. The collection
launcher now supports the same flag and records its distinct mode. Firmware
quantization means a zero request is not exactly zero physical torque.
The wrist must remain supported throughout. `--tx-trace PATH.npz` captures
the packed mode/MIT frames passed to the SDK transport, retaining the start
and a bounded tail without disk writes during the loop. It preserves send
exceptions and sends no extra frames. Successful sends do not prove motor
application. Effective requested gains and position/velocity references are
also recorded in the controller NPZ. A virtual-transport test exercises the
installed v183 encoder and verifies MIT flags, zero kp, the J6 zero-request
encodings and frame CRCs.

**Supported passive-J6 check passed, 2026-11-22:** retry session
`data_kinesthetic_j6_check_20261122/sessions/20261123T015722Z-6e7be1ae/`
ran ACTIVE for 118.065 s. The operator confirmed assist at target, that the
arm moved freely, and no issues after target. Its only freeze was `operator
quit`, followed by supported disable; fresh feedback then confirmed all six
drives off. Effective requested kp was zero on all joints throughout ACTIVE;
J6's requested kp, kd and feed-forward remained zero in every recorded state.
The bounded transport trace retains 132,000 of 294,125 packed sends (the
middle is deliberately omitted); retained frames have no send failures or
MIT CRC errors, and their mode frames consistently select CAN/MIT/0xAD.
J6's wire values match the v183 zero-request encodings. The log reports
4 overruns in 24,510 cycles (0.0163%), below the warning threshold; this
counter does not locate them within the run. Evidence, finalized control
log, trace and `diagnostic_summary.json` are saved in the session folder.

The first passive attempt froze on `runaway J4` at 0.225 s during ramp-in;
it never reached ACTIVE. Its log shows genuine motion while gravity torque
was still near zero. Full physical support through mode switching and the
four-second ramp is essential; the retry kept the original guards and gains.
This passes the supported free-wrist functional check, not the full-power
gate (J1/J2 remain capped) or a clean training pilot. Next collect pilot
episodes with `--passive-j6`, aligned to the task's recorded start, and resolve the
hand-limit findings and verify synchronization after the USB-latency change
before proceeding to Part 2.

At the operator's request, the collection launcher now supplies
`--j5-kd 0.005` to lower only J5's MIT damping during ramp-in and ACTIVE.
Other active-joint MIT gains, software damping, brake/hold gains, and the
passive-J6 zero requests are preserved. The override and derived damping
vector are saved in the session command/runtime metadata. This requested
0.005 maps to approximately 0.003663 in the installed v183 encoding; the
earlier successful hardware check used the previous J5 request of 0.01.
The subsequent camera pilot below exercised the 0.005 request on hardware,
with no operator-reported behavior issues. This does not clear that episode's
separate quality findings.

**Camera pilot with passive J6 and J5 Kd=0.005, 2026-11-22:**
`data_trimorph_kinesthetic/raw/episode_000000` saved 23.495 s via the foot
pedal: 4,690 Piper samples at 199.6 Hz (maximum gap 10.9 ms), 1,175 hand
samples at 50 Hz, and 705 camera frames at 30 Hz with no drops. All recorded
drive/status samples were healthy. The operator reported no issues during
recording and completed the supported stop; all six drives were then confirmed
disabled. The saved episode and FK processing demonstrate the acquisition
path, but this episode remains flagged for quality review:

- The first pose differed from the named start by up to 119.94 degrees, and
  no recorded sample met its all-six-joint tolerance. At collection time,
  the large green ARM health box could be mistaken for start readiness;
  the separate guide's angle calculation correctly reports that these poses
  are not at start. The monitor now makes START readiness the primary box
  and a filled red/green banner; feedback health is secondary text.
- J1 current-derived torque was near the cap for 20.454/23.490 observed
  seconds. J5 approached within 0.43 degrees of the configured limit.
- Motor 8 exceeded its +/-0.01 rad YAML limits in 969/1,175 samples; several
  other hand motors had smaller limit overshoots. Raw values and YAML limits
  are preserved for review.
- The USB adapter was still at 16 ms. Hand read intervals alternated near
  16/32 ms, and 106/703 camera frames were more than 10 ms from a hand sample
  (maximum 15.74 ms). Camera-to-Piper timing was within 2.15 ms.
- FK agreed with firmware within 1.22 mm and 0.245 degrees. Processing
  retained best-effort synchronization and does not clear the warnings.

Reports and finalized control evidence are stored under
`data_trimorph_kinesthetic/sessions/20261123T020934Z-80175bff/`.
After the supported stop, the operator set the USB latency timer with sudo,
and readback confirmed `1` ms on 2026-11-22. The START display clarification
was then used for the repeat below. Episode 000000 remains diagnostic.

**Repeat camera pilot after the 1 ms USB change, 2026-11-22:**
`data_trimorph_kinesthetic/raw/episode_000001` saved 28.780 s: 5,756 arm samples
at 200 Hz (maximum gap 6.1 ms), 1,439 hand samples at 50 Hz, and 864 camera
frames at 30 Hz with no drops. All recorded arm/status samples were healthy,
with no status transitions. Hand intervals improved to median 19.99 ms,
p95 21.21 ms and maximum 22.45 ms. Camera-to-hand timing improved to
0/862 retained frames beyond 10 ms, with maximum 8.56 ms, versus 106/703 and
15.74 ms in episode 000000. Camera-to-CAN-receipt timing stayed within 2.69 ms.
This verifies the synchronization improvement for this recording.

The episode is still diagnostic: its first pose differed from the legacy
`kinesthetic` target by up to 110.03 degrees; no sample met that target's
all-six-joint tolerance. J5 crossed its SDK limit in 105 samples and was within
3 degrees in another 369 (minimum margin -0.27 degrees). J1 current-derived
torque was near the cap for 22.375/28.775 observed seconds. Hand motor 8 was
outside its YAML limits in 1,406/1,439 samples; motors 0/4/10/11 also exceeded
limits. The read-only report is saved under
`sessions/20261123T021752Z-be605412/episode_000001_quality.txt`.

The operator reported that the legacy start was uncomfortable, leading to the
per-task definition workflow above. Hand-limit, arm-limit and cap findings
remain unresolved before training or replay.

**Task start captured, 2026-11-22:**
The 6.772-second `data_trimorph_kinesthetic/raw/episode_000002` is explicitly a
`start_definition` clip. Offline extraction saved
`config/kinesthetic_starts/multi_pick_and_place.json`, SHA-256
`3600463715b33008766c932ebf463d4ff1a08d987059dfc5f23e1f139069f605`.
Its measured arm target is
`[-47.622, 101.204, -44.759, -58.979, -57.856, -53.219]` degrees. The selected
arm and hand samples are 4.826 ms apart. The calibrated nominal arm check
passes with J1 +5.935/J2 -2.915 N*m and all joints within limits. J6's model
gravity demand is about +0.149 N*m; the passive profile still requests zero
J6 torque and needs wrist support.

The profile preserves all measured hand values and three unresolved warnings:
motor 4 is +0.008524 rad above its zero upper limit; motor 7 is +0.020796 rad
above its zero upper limit; motor 8 is -1.586758 rad against +/-0.01 rad limits.
No values were clipped and no hand limits were changed. The arm target can
guide collection for this task, but the recorded hand target is not cleared
for rollout. The definition clip is not a clean training demonstration.

**Startup drop reported, 2026-11-22 — do not repeat startup unchanged:**
Session `20261123T032509Z-bab14d8d` froze on `runaway J4` at 0.265 s,
before ramp-in completed. The recorded pre-freeze joints stayed at least
17.3 degrees inside their model limits, outside the 8.59-degree repulsion
band. J1/J2 moved about 18 degrees in that interval. The first J1 command
was only 0.0066 N*m against a nominal gravity demand of 5.27 N*m.

The startup sequence switches to zero-stiffness MIT, waits 0.3 s before
streaming commands, then ramps gravity support from zero over four seconds.
The evidence supports loss of support during that handover, rather than a
joint-limit trigger. Logging starts after the mode-switch wait, so it does
not capture the entire transition or prove delivered motor torque. A target
pose passing the static torque check does not validate this transition.
Review the load-transfer sequence before another supported hardware trial;
do not bypass the watchdog or abruptly step full torque in as a workaround.
The original log and `startup_incident_report.json` are preserved in the
session folder. The end-effector readiness display does not command motion.

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
3. 1.4 to 1.7 — implemented; offline verification complete.
4. Pilot collection of a few episodes — first episode flagged; resolve the findings above and retry before Part 2.
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
- **End pose in MIT mode.** Verified streaming on firmware `S-V1.8-5` in the 2026-11-22 bench (1.3). The planned FK poses from joints don't depend on this firmware stream.
