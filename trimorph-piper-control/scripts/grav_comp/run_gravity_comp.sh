#!/usr/bin/env bash
# Guided gravity_comp bring-up (midas-piper-control stages 0-6) for the Piper with the TRIMI hand.
#
# Runs the stage scripts in order, stops before each one, and tells you when to take the hand off
# (stages 0-3 use the bare-arm model, no hand guess) and when to mount it (stage 4a onward, where
# the hand's mass and CoM are calibrated from data). Each stage is the stage script itself
# (python -m gravity_comp.<stage>) with its own Enter-to-arm prompt and SPACE / Ctrl-C damp-brake.
# This wrapper never kills a running stage: on Ctrl-C, bash waits for the stage to finish its
# brake -> freeze-hold sequence before stopping.
#
# Progress lives in the calibration file, so the script starts at the first stage that has not
# passed yet. Run with --help for options.

set -uo pipefail

STEPS=(probe workspace feedback gravity j1 j2 j3 payload j5 j4 j6 friction drag)

declare -A TITLE=(
    [probe]="Stage 0    probe: firmware, rest noise, damp-brake drill"
    [workspace]="Stage 0.5  workspace keep-out plane"
    [feedback]="Stage 1    torque feedback scale (beta)"
    [gravity]="Stage 2    gravity vector"
    [j1]="Stage 3    torque delivery J1 (MIT k-ramp)"
    [j2]="Stage 3    torque delivery J2 (MIT k-ramp)"
    [j3]="Stage 3    torque delivery J3 (MIT k-ramp)"
    [payload]="Stage 4a   hand mass + CoM"
    [j5]="Stage 4b   torque delivery J5 (MIT k-ramp)"
    [j4]="Stage 4b   torque delivery J4 (MIT k-ramp)"
    [j6]="Stage 4c   J6 breakaway (MIT ramp)"
    [friction]="Stage 5    friction, all joints (MIT ramps)"
    [drag]="Stage 6    drag mode + acceptance tests A1-A5"
)
declare -A MINUTES=([probe]=10 [workspace]=2 [feedback]=25 [gravity]=30 [j1]=12 [j2]=7 [j3]=12
                    [payload]=25 [j5]=12 [j4]=12 [j6]=10 [friction]=30 [drag]=20)
declare -A HAND=([probe]=off [workspace]=off [feedback]=off [gravity]=off [j1]=off [j2]=off
                 [j3]=off [payload]=on [j5]=on [j4]=on [j6]=on [friction]=on [drag]=on)
# Behind-the-base notes: scripts/grav_comp clearance results (2026-11-21, see README.md).
declare -A NOTE=(
    [probe]="Press SPACE during the small J6 torque pulse when it tells you to (brake drill)."
    [j1]="Don't touch the arm. The 'J1 low-g' pose won't move on the bare arm (~0.12 N*m gravity vs ~0.6 N*m friction): no-motion-at-k-max there is expected. Only 'J1 high-g' counts; it needs 2 trials that stay inside 0.10 rad."
    [j3]="Runs with a 0.2 rad catch radius: J3 creeps after it breaks free. J3 swings the arm up to ~34 cm behind the base's mounting face at 1.25-1.9 m height. Keep everyone off the west side of the base."
    [j5]="J5 holds between k~1.14 and ~1.40 with this hand. --skip-nudge: the spring-only nudge hold drops the wrist onto its hard stop (~0.32 rad past the pose)."
    [j4]="J4's hard stop is only ~0.195 rad past the J4 probe pose, hence the 0.15 rad catch radius. Reaches behind the base and ENDS behind it (J4-): be ready to catch the arm there."
    [friction]="J2 reaches behind the base. Keep everyone off the west side of the base."
    [drag]="UNDERPOWERED: J1/J2 torque is capped at 7.2 N*m, so the arm is only held where they need less (around quest_0, home_4a, home_cube_newhw, home_unscrew1). Around home_4b/4c/4d/m3 it sags: keep a hand on it. Start in a held pose; ramp-in takes 4 s from zero, support the arm until it carries its own weight."
)

default_midas_repo() {
    local d
    for d in "$HOME/midas_hand/midas-piper-control" "$HOME/workspace/midas-piper-control"; do
        [[ -d "$d/gravity_comp" ]] && { echo "$d"; return; }
    done
    echo "$HOME/midas_hand/midas-piper-control"
}

MIDAS_REPO="$(default_midas_repo)"
# The calibration is versioned in the trimi repo, next to this script.
CALIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/calib/robot1_trimi.json"
CHANNEL="can0"
TABLE_DEPTH="0.49"
HAND_MASS=""
FROM=""
LIST=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Runs gravity_comp stages 0-6 in order for the Piper + TRIMI hand, stopping before each stage and
telling you when to take the hand off and when to mount it. Starts at the first stage the
calibration file has not passed yet.

Options:
  --midas-repo DIR   midas-piper-control checkout (default: $MIDAS_REPO)
  --calib PATH       calibration file, absolute or relative to the midas repo
                     (default: $CALIB)
  --channel NAME     CAN interface (default: $CHANNEL)
  --table-depth M    stage 0.5 keep-out plane below the base, = base above desk (default: $TABLE_DEPTH)
  --hand-mass KG     scale mass of everything on link6, for stage 4a (asked for if not given)
  --from STEP        start at STEP instead of the first unfinished one
  --list             show the steps and what the calibration file has passed, then exit
  --dry-run          go through the prompts but print the commands instead of running them
  -h, --help         show this help

Steps: ${STEPS[*]}
  (aliases: 0 0.5 1 2 3 4a 4b 4c 5 6, and mount = payload)
EOF
}

die() { echo "error: $*" >&2; exit 1; }

ask() {  # ask PROMPT -> ANS
    if ! read -r -p "$1 " ANS; then echo; exit 130; fi
}

resolve_step() {
    case "$1" in
        0) echo probe ;; 0.5) echo workspace ;; 1) echo feedback ;; 2) echo gravity ;;
        3) echo j1 ;; 4a) echo payload ;; mount) echo payload ;; 4b) echo j5 ;;
        4c) echo j6 ;; 5) echo friction ;; 6) echo drag ;;
        *) local s; for s in "${STEPS[@]}"; do [[ $s == "$1" ]] && { echo "$s"; return; }; done
           return 1 ;;
    esac
}

step_index() {
    local i
    for i in "${!STEPS[@]}"; do [[ ${STEPS[$i]} == "$1" ]] && { echo "$i"; return; }; done
    return 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --midas-repo)  MIDAS_REPO="${2:?--midas-repo needs a value}"; shift 2 ;;
        --calib)       CALIB="${2:?--calib needs a value}"; shift 2 ;;
        --channel)     CHANNEL="${2:?--channel needs a value}"; shift 2 ;;
        --table-depth) TABLE_DEPTH="${2:?--table-depth needs a value}"; shift 2 ;;
        --hand-mass)   HAND_MASS="${2:?--hand-mass needs a value}"; shift 2 ;;
        --from)        FROM="$(resolve_step "${2:?--from needs a value}")" || die "unknown step '$2'"; shift 2 ;;
        --list)        LIST=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             usage >&2; die "unknown option: $1" ;;
    esac
done

valid_mass() { [[ $1 =~ ^[0-9]*\.?[0-9]+$ ]] && awk -v m="$1" 'BEGIN { exit !(m > 0.05 && m < 5) }'; }
[[ -z $HAND_MASS ]] || valid_mass "$HAND_MASS" || die "--hand-mass '$HAND_MASS' is not a mass in kg"
if [[ $CALIB == /* ]]; then CALIB_SHOWN=$CALIB; else CALIB_SHOWN=$MIDAS_REPO/$CALIB; fi

# ------------------------------------------------------------------ preflight

[[ -d "$MIDAS_REPO/gravity_comp" ]] || die "no gravity_comp/ in $MIDAS_REPO (pass --midas-repo)"
cd "$MIDAS_REPO" || die "cannot cd to $MIDAS_REPO"
PY="$MIDAS_REPO/.venv/bin/python"
[[ -x $PY ]] || PY="$(command -v python3)"
"$PY" -c "import mujoco, numpy" 2>/dev/null \
    || die "$PY has no mujoco/numpy (the midas repo's .venv should have them)"

build_cmd() {  # build_cmd STEP -> CMD
    local mass="${HAND_MASS:-<kg>}"
    local run=("$PY" -m)
    case "$1" in
        probe)     CMD=("${run[@]}" gravity_comp.probe --channel "$CHANNEL" --calib "$CALIB") ;;
        workspace) CMD=("${run[@]}" gravity_comp.set_workspace --calib "$CALIB" --table-depth "$TABLE_DEPTH") ;;
        feedback)  CMD=("${run[@]}" gravity_comp.characterize_feedback --no-hand --channel "$CHANNEL" --calib "$CALIB") ;;
        gravity)   CMD=("${run[@]}" gravity_comp.calibrate --stage gravity --no-hand --channel "$CHANNEL" --calib "$CALIB") ;;
        # J1: only the high-g pose can break away on the bare arm, and J1 is sticky (it can slide past
        # the 0.10 rad catch radius at breakaway), so give it 4 trials per pose instead of 2.
        j1)        CMD=("${run[@]}" gravity_comp.verify_delivery --joint 1 --no-hand --trials 4 --channel "$CHANNEL" --calib "$CALIB") ;;
        j2)        CMD=("${run[@]}" gravity_comp.verify_delivery --joint 2 --no-hand --channel "$CHANNEL" --calib "$CALIB") ;;
        # J3 sticks (~1.1 N*m breakaway) and keeps creeping at k_up after it breaks free, so it is
        # already 0.05-0.07 rad out when motion is confirmed and crosses 0.10 rad while settling.
        # 0.20 rad passes the stage's own workspace precheck (>= 515 mm to the table) and stays
        # inside measure_friction's 0.30 rad; 3 trials per pose in case the low-g pose cannot sink.
        j3)        CMD=("${run[@]}" gravity_comp.verify_delivery --joint 3 --no-hand --catch-rad 0.2 --trials 3 --channel "$CHANNEL" --calib "$CALIB") ;;
        payload)   CMD=("${run[@]}" gravity_comp.calibrate --stage payload --hand-mass "$mass" --channel "$CHANNEL" --calib "$CALIB") ;;
        # 4b with the TRIMI hand (measured 2026-11-22), J5 and J4 run separately because they need
        # different catch radii. Both: position mode leaves the probe joint ~0.13 rad short of its
        # pose; J5 is friction-held between k~1.14 and ~1.40, so ramps start at k 1.25; probe kd 2.0
        # damps the breakaway snap (its bias on k_up/k_down is opposite and cancels in alpha);
        # --skip-nudge because the spring-only nudge hold drops the wrist onto its hard stop.
        # J5: hard stop ~0.32 rad past the pose -> catch 0.2. J4: hard stop only ~0.195 rad past
        # the pose -> catch 0.15, so a joint resting on the stop aborts at once instead of reading
        # as "no droop" (that is what produced the false J4 sign flip with catch 0.2).
        j5)        CMD=("${run[@]}" gravity_comp.verify_delivery --joint 5 --with-hand --catch-rad 0.2 --k-min 1.25 --probe-kd 2.0 --trials 3 --skip-nudge --channel "$CHANNEL" --calib "$CALIB") ;;
        j4)        CMD=("${run[@]}" gravity_comp.verify_delivery --joint 4 --with-hand --catch-rad 0.15 --k-min 1.25 --probe-kd 2.0 --trials 3 --skip-nudge --channel "$CHANNEL" --calib "$CALIB") ;;
        # 4c: the settings of the only July run that passed on this arm (after 6 failures).
        j6)        CMD=("${run[@]}" gravity_comp.verify_delivery --j6 --with-hand --j6-kd 1.0 --j6-abort-v 0.5 --trials 3 --channel "$CHANNEL" --calib "$CALIB") ;;
        friction)  CMD=("${run[@]}" gravity_comp.measure_friction --all --with-hand --channel "$CHANNEL" --calib "$CALIB") ;;
        # 6: friction's asymmetry/spread gates fail on this arm (real direction-dependent friction;
        # the section is saved forced, weaker direction stored for assist sizing), so drag mode needs
        # --force. With the v183 driver the firmware caps torque-mode commands at 8 N*m, while holding
        # the arm + TRIMI everywhere needs up to ~15 N*m on J1/J2, so DragConfig only validates with
        # --allow-underpowered: the arm is held only where J1/J2 need <= 7.2 N*m.
        drag)      CMD=("${run[@]}" gravity_comp.drag_mode --force --allow-underpowered --channel "$CHANNEL" --calib "$CALIB") ;;
    esac
}

passed_steps() {  # the steps the calibration file records as passed
    "$PY" - "$CALIB" <<'EOF'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}

def ok(name):
    section = data.get(name)
    return isinstance(section, dict) and section.get("gates_passed") is True

measured = (data.get("delivery") or {}).get("measured") or [False] * 6
passed = {
    "probe": ok("firmware") and ok("noise"),
    "workspace": ok("workspace"),
    "feedback": ok("feedback"),
    "gravity": ok("gravity"),
    "j1": measured[0], "j2": measured[1], "j3": measured[2],
    "payload": ok("payload"),
    "j5": measured[4],
    "j4": measured[3],
    "j6": measured[5],
    # A forced friction section (failed asymmetry/spread, accepted by drag mode's --force) counts.
    "friction": ok("friction") or bool((data.get("friction") or {}).get("forced")),
    "drag": ok("runtime"),
}
print(" ".join(step for step, done in passed.items() if done))
EOF
}

PASSED=" $(passed_steps) "
if [[ -n $FROM ]]; then
    START=$(step_index "$FROM")
else
    START=${#STEPS[@]}
    for i in "${!STEPS[@]}"; do
        [[ $PASSED == *" ${STEPS[$i]} "* ]] || { START=$i; break; }
    done
fi

show_steps() {
    local i step mark arrow
    echo "Calibration file: $CALIB_SHOWN"
    echo
    for i in "${!STEPS[@]}"; do
        step=${STEPS[$i]}
        [[ $step == probe ]]   && echo "        -- hand OFF: bare link6 flange --"
        [[ $step == payload ]] && echo "        -- mount the TRIMI hand --"
        mark="    "; [[ $PASSED == *" $step "* ]] && mark="pass"
        arrow="  ";    [[ $i == "$START" ]] && arrow="->"
        printf ' %s [%s] %-9s %s (~%s min)\n' "$arrow" "$mark" "$step" "${TITLE[$step]}" "${MINUTES[$step]}"
    done
    echo
}

show_steps
if (( LIST )); then exit 0; fi
if (( START >= ${#STEPS[@]} )); then
    echo "Every stage has passed. To run drag mode again: $(basename "$0") --from drag"
    exit 0
fi

if (( ! DRY_RUN )); then
    ip -brief link show "$CHANNEL" 2>/dev/null | grep -qw UP \
        || die "CAN interface $CHANNEL is not up: run trimorph-piper-control/setup_arm_can.sh"
fi
echo "Offline unit tests (gravity_comp/tests)..."
if ! TEST_OUT=$("$PY" -m pytest gravity_comp/tests -q -x 2>&1); then
    echo "$TEST_OUT" | tail -20
    die "gravity_comp unit tests fail: fix them before any hardware work"
fi
echo "  $(echo "$TEST_OUT" | tail -1)"

cat <<'EOF'

  SAFETY (gravity_comp README section 3)
  - Someone stands at the arm with a hand free to catch it during every MIT torque stage.
  - Each stage prints what it will do and waits for Enter before it enables torque: read it first.
  - SPACE or Ctrl-C = damp-brake, then freeze-hold. The hardware e-stop cuts power and the ARM
    FALLS: last resort only.
  - Disable only from freeze-hold, at a supported pose, with your hand on the arm.
  - Support the arm across every position -> MIT mode switch.

EOF

ask "Start at '${STEPS[$START]}'? [Enter] yes, or type another step name, q quits:"
case $ANS in
    "") ;;
    q|Q) exit 0 ;;
    *) step=$(resolve_step "$ANS") || die "unknown step '$ANS'"; START=$(step_index "$step") ;;
esac

# ------------------------------------------------------------------ hand changes

HAND_STATE=unknown

hand_off() {
    cat <<'EOF'

==> Take the TRIMI hand OFF.
    Remove the hand, flange adapter, Quest controller and camera down to the bare link6 flange.
    Stages 0-3 use the bare-arm model, so anything left on the flange is unmodelled load.
    Support the arm while you do it.
EOF
    while :; do
        ask "    Type 'off' when the flange is bare (q quits):"
        case $ANS in off) break ;; q|Q) exit 0 ;; esac
    done
    HAND_STATE=off
}

hand_on() {
    if [[ $HAND_STATE == off ]]; then
        cat <<'EOF'

==> Stages 0-3 done. MOUNT THE TRIMI HAND now.
    Mount it with everything it carries in use: flange adapter, Quest controller, camera.
    Dress the cables so they don't pull on the wrist: cable tension reads as extra payload.
    Support the arm while you do it.
EOF
    else
        cat <<'EOF'

==> Check the TRIMI hand is mounted as it was for stage 4a
    (flange adapter, Quest controller, camera, cables dressed).
EOF
    fi
    while :; do
        ask "    Type 'mounted' when ready (q quits):"
        case $ANS in mounted) break ;; q|Q) exit 0 ;; esac
    done
    HAND_STATE=on
}

payload_setup() {
    cat <<'NOTE'

    Stage 4a fits the hand's mass and CoM from the joint torques ("mass free-solve" in its
    output). The mass you enter is what that fit is checked against (within 10%), and the CoM is
    checked against the MIDAS hand's (gravity_comp/model.py HAND_COM_PRIOR_LINK6_M).
    No scale: enter a rough guess. It changes neither the motion (position mode) nor the fit.
    If 4a then refuses only on mass_consistency / com_dev_*, quit and re-solve from the saved
    capture with the arm's own mass (seconds, no arm time), then run this script again:
      .venv/bin/python -m gravity_comp.resolve_offline --what payload --write \
          --npz gravity_comp/calib/piper-1_payload_<time>.npz --mass-fixed <free-solve kg>
NOTE
    while [[ -z $HAND_MASS ]]; do
        ask "    Mass of everything now on link6 (hand + adapter + controller + camera), kg:"
        if valid_mass "$ANS"; then HAND_MASS=$ANS; else echo "    '$ANS' is not a mass in kg"; fi
    done
}

# ------------------------------------------------------------------ run

CURRENT=""
trap 'echo; echo "Stopped${CURRENT:+ at $CURRENT}. Resume with: $(basename "$0")${CURRENT:+ --from $CURRENT}"; exit 130' INT

N=${#STEPS[@]}
for (( i = START; i < N; i++ )); do
    step=${STEPS[$i]}
    CURRENT=$step
    if [[ ${HAND[$step]} != "$HAND_STATE" ]]; then
        if [[ ${HAND[$step]} == off ]]; then hand_off; else hand_on; fi
    fi
    [[ $step == payload ]] && payload_setup

    build_cmd "$step"
    echo
    echo "==> [$((i + 1))/$N] ${TITLE[$step]}   ~${MINUTES[$step]} min, hand ${HAND[$step]^^}"
    [[ $PASSED == *" $step "* ]] && echo "    (already passed in the calibration file: running it again overwrites it)"
    [[ -n ${NOTE[$step]:-} ]] && echo "    ${NOTE[$step]}"
    echo "    \$ ${CMD[*]}"
    ask "    [Enter] run   s skip   q quit:"
    case $ANS in
        s|S) echo "    skipped"; continue ;;
        q|Q) echo "Resume with: $(basename "$0") --from $step"; exit 0 ;;
    esac

    while :; do
        if (( DRY_RUN )); then echo "    (dry run: not running)"; break; fi
        "${CMD[@]}"
        rc=$?
        if (( rc == 0 )); then echo; echo "    $step passed"; break; fi
        echo
        echo "    $step stopped with exit code $rc (failed checks, missing prerequisite, or aborted)."
        ask "    [r] retry   [q] quit:"
        case $ANS in
            r|R) ;;
            *) echo "Resume with: $(basename "$0") --from $step"; exit "$rc" ;;
        esac
    done
done

echo
echo "All stages done. Calibration: $CALIB_SHOWN"
echo "Drag mode again later: $(basename "$0") --from drag"
