# Gravity-comp motion checks (TRIMI hand on the Piper)

Before running midas-piper-control's `gravity_comp` bring-up (branch
`norman/piper-midas-gravity-comp`) with the TRIMI hand, these scripts replay every stage's arm
motion in the rig layout and check what it comes close to: the desk, the old cart and post, the
floor, the mount plate, the arm itself, the base's mounting face and the standing wearer.

| File | What it does |
|---|---|
| `grav_comp_clearance.py` | Clearance table per pose for every stage (S1–S6); `--render` saves pictures of flagged poses |
| `grav_comp_video.py` | The same motions as videos: one MP4 per stage plus `all_stages.mp4` / `.webm` |
| `held_joint_drift.py` | The torque stages with the spring-held joints drifted to the 0.08 rad abort limit |
| `run_gravity_comp.sh` | Runs the real bring-up on the arm (stages 0–6) as one guided session: stops before each stage, says when to take the hand off (0–3) and mount it (4a on), resumes from the calibration file |
| `calib/robot1_trimi.json` | The Piper + TRIMI calibration from the 2026-11-22 bring-up, which `run_gravity_comp.sh` reads and writes (see the notes at the end) |
| `tests/` | `python -m pytest scripts/grav_comp/tests` |

## Setup

- A midas-piper-control checkout on `norman/piper-midas-gravity-comp`: the poses, timing and
  guard come from its `gravity_comp/`. Default `~/workspace/midas-piper-control`, else pass
  `--midas-repo`.
- The trimorph-piper-control workspace sim (`scripts/workspace/analysis.py`, `assets/piper`):
  the Piper URDF and meshes, the TRIMI hand hulls and the old cart and post come from it.
  Default `~/trimorph-piper-control`, else pass `--workspace-repo`.
- `pip install mujoco scipy pillow`, plus `imageio-ffmpeg` for the videos.
- Videos use MuJoCo's default GLFW context (needs a display). On a headless machine set
  `MUJOCO_GL=osmesa` (software, much slower).

## Usage

```bash
python3 scripts/grav_comp/grav_comp_clearance.py --viscous            # all stages, summary at the end
python3 scripts/grav_comp/grav_comp_clearance.py --stages S3 S5 --render
python3 scripts/grav_comp/grav_comp_video.py --viscous                # output/grav_comp_clearance/video/
python3 scripts/grav_comp/held_joint_drift.py --viscous

# On the arm (CAN up, someone at the arm). Calibration: scripts/grav_comp/calib/robot1_trimi.json
scripts/grav_comp/run_gravity_comp.sh --list        # what has passed, where it would start
scripts/grav_comp/run_gravity_comp.sh               # guided run from the first unfinished stage
scripts/grav_comp/run_gravity_comp.sh --from 4a --hand-mass 0.9
```

Common options: `--base-above-desk` (0.49 m as built), `--desk-height` (0.75 m),
`--start-deg` (the pose the arm rests in when a stage starts, default zero), `--hand-turn-deg`.
Output goes to `output/` (git-ignored).

Layout (config 2 of the trimorph-piper-control workspace sim): base at (0, 0, 1.24) with roll
90°, yaw 90° (J1 axis east, over the desk); desk x 0.021–0.521, y −0.058–0.918, top 0.75 m;
wearer at (−0.255, 0.225), facing 51°, 1.7 m. World axes: x east, y north, z up.

Columns (mm): `desk`; `other` = cart, post, floor, mount plate; `hand-arm`; `face` = how far
the arm stays in front of the base's mounting face (negative = behind it, where the mount
structure is, which the sim does not model); `person` = the config-2 wearer; `guard` = the
clearance gravity_comp's own guard computes (table plane only).

## How close the replay is to gravity_comp

- **S1, S2, S4a:** the same. Same poses and order, `capture_static_sample` (staging point
  below, half-speed approach, settle and average, then from above), return to the start pose.
- **S3, S4b, S4c, S5:** the position-mode path is the same. The probe joint is drawn out to its
  catch radius both ways (0.10 rad, S5 0.30 rad), the bound where gravity_comp aborts; on the
  arm it usually moves less. `held_joint_drift.py` covers the held joints; braking past the
  radius is not modelled. The video's time for these stages is not real (the ramps are slower).
- **S5 `--viscous`:** each joint's two sweeps (0.10, 0.25 rad/s) follow its friction poses, as
  in measure_friction.
- **S6:** drag_mode moves only by hand; the replay joins the 8 A1 poses with straight lines.
- Every stage starts from `--start-deg`. On the rig, S3–S6 end at their last pose and the arm
  goes limp after the confirm, so the next stage starts wherever the operator left it.

## Results (2026-11-21: base 0.49 m above a 0.75 m desk, start pose zero)

- Desk, cart, post, floor, mount plate: clear in every stage. Desk ≥ 97 mm (S4a); mount plate
  ≥ 67 mm, 30 mm with the held joints drifted.
- Hand to arm: ≥ 28 mm, closest at the folded zero start pose.
- S3 `--joint 3`, S4b `--wrist` and S5 J2 (with `--viscous` also its sweeps) reach up to 34 cm
  behind the base's mounting face at 1.25–1.9 m height, through where the config-2 wearer
  stands; 6 cm more with the held joints drifted. Check the real mount structure there, keep
  people off the west side of the base, and note that S4b ends behind the base (J4−), where the
  operator has to catch the arm.
- gravity_comp's own guard never trips: it checks a table plane only and does not see the
  hand's sides or anything behind the base.

## Calibration of 2026-11-22 (`calib/robot1_trimi.json`)

Firmware S-V1.8-5 with the `v183` driver, which caps torque-mode commands at 8 N·m on every
joint. All stage settings below are built into `run_gravity_comp.sh`.

- **Delivery alpha J1–J6:** 0.988, 0.980, 1.037, 0.793, 0.740, 0.740 (J6 set equal to J5 by
  design). The wrist delivers only about 75–80% of the commanded torque.
- **Payload:** 0.823 kg, CoM (19.0, 11.7, 116.7) mm in link6, from the arm's own fit (no scale
  reading). The fit's rms (0.140 N·m) is over stage 4a's 0.10 limit, so it was saved with
  `resolve_offline --rms-gate 0.15`, stage 2's limit. It is valid for the hand as mounted that
  night; mounting it differently means redoing stage 4a.
- **Friction:** J1 0.669, J2 0.784, J3 0.857, J5 0.086 N·m (weaker direction). J4 and J6 have
  no data (the wrist trips the speed abort), so drag mode gives them no friction assist. J1, J2
  and J5 fail the asymmetry/spread checks, so the section is saved forced and drag mode needs
  `--force`.
- **Drag mode is underpowered on J1/J2:** holding the arm everywhere needs up to ~15 N·m there,
  against the 7.2 N·m usable cap, so it runs with `--allow-underpowered`. The arm holds around
  the `quest_0`, `home_4a`, `home_cube_newhw` and `home_unscrew1` start poses and sags around
  `home_4b`/`4c`/`4d`/`m3`. Firmware 1.8-8 or newer (16 N·m) would cover all start poses.
- **Wrist hard stops** at about J4 ±1.79 and J5 ±1.31 rad, past the URDF limits. Stage 4b's J4
  probe pose sits only 0.195 rad from J4's, hence its 0.15 rad catch radius.
- **Needs these midas-piper-control changes** (not upstream as of this date): the bare-arm
  (`--no-hand`) stages zero the model's 0.7 kg placeholder payload, and `resolve_offline`
  takes `--rms-gate`.
