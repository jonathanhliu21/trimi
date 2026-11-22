#!/usr/bin/env python3
"""Compare a replay's calibrated Vicon flange position with its target and Piper feedback."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_collection.postprocess_dataset import convert_mocap, controller_rotation, frame_clock, smooth_poses
from data_collection.replay_episode import load_hand_stream, load_piper_stream, replay_interval, sample_stream
from scripts.teleop.quest_replay import calibration_origin
from scripts.calib.get_coord_transform import tracked_point_offset


def read_numeric(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
            for key in rows[0]}


def metrics(error):
    return {axis: {'bias_mm': float(np.mean(v)), 'mae_mm': float(np.mean(abs(v))),
                   'rmse_mm': float(np.sqrt(np.mean(v ** 2))), 'max_abs_mm': float(np.max(abs(v)))}
            for axis, v in zip('xyz', error.T * 1000)}


def legacy_mocap(path, calibration, topic, *, smooth_s=0, use_frame_clock=True):
    """Original similarity calibration used by the screw dataset, including its lever arm."""
    if calibration.get('tracker') != 'mocap':
        raise ValueError('Expected a mocap calibration')
    with Path(path).open() as stream:
        rows = [r for r in csv.DictReader(stream)
                if r['topic'] == topic and r['message_type'] == 'vicon_receiver/msg/Position']
    poses = np.array([[float(r[k]) for k in ('x_trans','y_trans','z_trans','x_rot','y_rot','z_rot','w')]
                      for r in rows])
    if len(rows) < 3:
        raise ValueError('Need at least three matching Vicon poses')
    ns = np.array([int(r['monotonic_ns']) for r in rows], dtype=np.int64)
    valid = np.isfinite(poses).all(axis=1) & (np.abs(poses[:, :3]).sum(axis=1)>0) & (np.linalg.norm(poses[:, 3:],axis=1)>.5)
    frames = np.array([int(r['frame_number']) for r in rows])
    valid &= np.r_[True, np.diff(frames)>0] & np.r_[True, np.diff(ns)>0]
    poses, ns, frames = poses[valid], ns[valid], frames[valid]
    if use_frame_clock:
        ns = frame_clock(frames, ns)
    positions, quat = smooth_poses(poses[:, :3]/1000, poses[:, 3:], np.median(np.diff(ns))/1e9, smooth_s)
    rotation = Rotation.from_matrix(calibration['orientation']['rotation_matrix']) * Rotation.from_quat(quat) * controller_rotation(calibration)
    origin, _ = calibration_origin(calibration)
    xyz = positions @ np.asarray(calibration['linear_matrix']).T + origin + rotation.apply(tracked_point_offset(calibration))
    return np.column_stack([xyz, rotation.as_rotvec()]), ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--episode', type=int, required=True)
    parser.add_argument('--positions', type=Path, required=True)
    parser.add_argument('--mocap', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--topic', default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    episode = f'episode_{args.episode:06d}'
    raw = args.dataset / 'raw' / episode
    processed = args.dataset / 'processed' / episode
    meta = json.loads((processed / 'metadata.json').read_text())
    if meta['pose_frame'] != 'piper_base':
        raise ValueError('This comparison requires absolute piper_base targets')
    calibration_path = Path(meta['calibration_json'])
    calibration_bytes = calibration_path.read_bytes()
    if hashlib.sha256(calibration_bytes).hexdigest() != meta['calibration_sha256']:
        raise ValueError('Calibration differs from the one used to process the episode')
    calibration = json.loads(calibration_bytes)
    log = read_numeric(args.positions)
    elapsed = log['elapsed_s']
    # Fit the replay's episode clock to host CLOCK_MONOTONIC using this run's telemetry.
    stamps = log['sample_monotonic_ns']
    if len(stamps) < 3 or np.any(np.diff(stamps) <= 0) or np.any(np.diff(elapsed) <= 0):
        raise ValueError('Need at least three increasing telemetry timestamps')
    origin_reference = stamps[0]
    speed, intercept = np.polyfit((stamps - origin_reference) / 1e9, elapsed, 1)
    origin_ns = int(round(origin_reference - intercept / speed * 1e9))
    hand = load_hand_stream(raw / 'leader_hand.csv')
    arm = load_piper_stream(processed / 'piper.csv')
    start, end = replay_interval(hand, arm)
    topic = args.topic or meta['mocap_topic']
    reconstruction_mm = None
    if not calibration.get('vicon_reference'):
        # Validate the older mapping against the already processed episode before using it.
        original, original_ns = legacy_mocap(raw / 'mocap.csv', calibration, meta['mocap_topic'],
                                              smooth_s=meta['mocap_smooth_s'], use_frame_clock=False)
        episode_origin = int(json.loads((raw / 'metadata.json').read_text())['started_monotonic_ns'])
        original_t = (original_ns-episode_origin)/1e9
        rebuilt = np.column_stack([np.interp(arm[0], original_t, original[:, i]) for i in range(3)])
        reconstruction_mm = float(np.max(abs(rebuilt-arm[1][:, :3]))*1000)
        if reconstruction_mm > .001:
            raise ValueError(f'Original mapping reconstruction differs by {reconstruction_mm:g} mm')
        mocap_poses, mocap_ns = legacy_mocap(args.mocap, calibration, topic)
    else:
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'mocap.csv').symlink_to(args.mocap.resolve())
            (temp / 'metadata.json').write_text(json.dumps({'started_monotonic_ns': origin_ns}))
            _, mocap_poses, mocap_ns, _, topic = convert_mocap(
                SimpleNamespace(path=temp, metadata_path=temp / 'metadata.json'), calibration, None,
                topic=topic, absolute=True, smooth_s=0)
    mocap_t = speed * ((mocap_ns - origin_ns) / 1e9)
    valid = (mocap_t >= 0) & (mocap_t <= end - start)
    mocap_t, mocap_poses = mocap_t[valid], mocap_poses[valid]
    if len(mocap_t) < 3:
        raise ValueError('No overlapping valid Vicon samples')
    target_mocap = np.array([sample_stream(*arm, start + t)[:3] for t in mocap_t])
    measured_mocap = mocap_poses[:, :3].astype(float)
    valid_log = (elapsed >= mocap_t[0]) & (elapsed <= mocap_t[-1])
    actual_piper = np.column_stack([log[f'piper_{axis}_m'] for axis in 'xyz'])[valid_log]
    log_t = elapsed[valid_log]
    target_log = np.array([sample_stream(*arm, start + t)[:3] for t in log_t])
    mocap_at_log = np.column_stack([np.interp(log_t, mocap_t, measured_mocap[:, i]) for i in range(3)])
    if not np.isfinite(actual_piper).all():
        raise ValueError('Missing Piper feedback in comparison interval')
    summary = {
        'episode': args.episode, 'topic': topic, 'calibration': str(calibration_path),
        'inferred_playback_speed': float(speed), 'episode_interval_s': [float(mocap_t[0]), float(mocap_t[-1])],
        'mocap_samples': len(mocap_t), 'piper_samples': len(log_t),
        'original_target_reconstruction_max_mm': reconstruction_mm,
        'error_convention': 'first named measurement minus second, in mm',
        'vicon_minus_target': metrics(measured_mocap - target_mocap),
        'piper_minus_target': metrics(actual_piper - target_log),
        'vicon_minus_piper': metrics(mocap_at_log - actual_piper),
        'notes': ['Vicon converted with the original calibration, including orientation-dependent marker-to-flange offset.',
                  'Same calibrated marker mounting required; no spatial reanchoring or offset removal.',
                  'Vicon frame clock reconstructed as in postprocessing; no pose smoothing applied.',
                  'Piper comparisons use sparse motor telemetry; Vicon-target comparison uses all valid frames.',
                  'Time alignment inferred from telemetry host stamps; individual read and network delays remain.'],
    }
    recorded_hand = np.array([sample_stream(*hand, start+t) for t in elapsed])
    hand_summary = []
    for i in range(13):
        target = log[f'motor_{i}_target_position_rad']
        actual = log[f'motor_{i}_actual_position_rad']
        servo_error = np.degrees(actual-target)
        modification = np.degrees(target-recorded_hand[:, i])
        hand_summary.append({'motor_id': i, 'tracking_mae_deg': float(np.mean(abs(servo_error))),
                             'tracking_max_deg': float(np.max(abs(servo_error))),
                             'target_vs_recording_mae_deg': float(np.mean(abs(modification))),
                             'actual_vs_recording_mae_deg': float(np.mean(abs(np.degrees(actual-recorded_hand[:, i]))))})
    summary['hand_motor_metrics'] = hand_summary
    (args.output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    np.savetxt(args.output / 'vicon_vs_target.csv',
               np.column_stack([mocap_t, target_mocap, measured_mocap, (measured_mocap-target_mocap)*1000]),
               delimiter=',', header='episode_elapsed_s,target_x_m,target_y_m,target_z_m,vicon_x_m,vicon_y_m,vicon_z_m,error_x_mm,error_y_mm,error_z_mm', comments='')
    np.savetxt(args.output / 'vicon_vs_piper.csv',
               np.column_stack([log_t, target_log, actual_piper, mocap_at_log]), delimiter=',',
               header='episode_elapsed_s,target_x_m,target_y_m,target_z_m,piper_x_m,piper_y_m,piper_z_m,vicon_x_m,vicon_y_m,vicon_z_m', comments='')
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-replay-mocap')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for i, axis in enumerate('XYZ'):
        ax = axes[i, 0]
        ax.plot(mocap_t, target_mocap[:, i]*1000, label='Episode target', color='black', lw=1.6)
        ax.plot(mocap_t, measured_mocap[:, i]*1000, label='Live Vicon', color='tab:orange', lw=1)
        ax.plot(log_t, actual_piper[:, i]*1000, '.-', label='Piper feedback', color='tab:blue', ms=3)
        ax.set_ylabel(f'{axis} position (mm)')
        ax = axes[i, 1]
        ax.plot(mocap_t, (measured_mocap-target_mocap)[:, i]*1000, label='Vicon − target', color='tab:orange')
        ax.plot(log_t, (actual_piper-target_log)[:, i]*1000, '.-', label='Piper − target', color='tab:blue')
        ax.plot(log_t, (mocap_at_log-actual_piper)[:, i]*1000, label='Vicon − Piper', color='tab:green')
        ax.axhline(0, color='black', lw=.7)
        ax.set_ylabel(f'{axis} error (mm)')
    for ax in axes.flat:
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Elapsed episode time (s; wall time is approximately twice this at 0.5×)')
    fig.suptitle(f'Episode {args.episode}: external Vicon measurement vs target and Piper feedback')
    fig.tight_layout()
    fig.savefig(args.output / 'position_comparison.png', dpi=170)
    plt.close(fig)
    labels = ['Thumb IP','Thumb MCP','Thumb CMC side','Thumb CMC roll','Index PIP','Index MCP','Index ABAD',
              'Middle PIP','Middle MCP','Middle ABAD','Ring PIP','Ring MCP','Ring ABAD']
    fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True)
    for i, ax in enumerate(axes.flat):
        if i >= 13:
            ax.set_visible(False)
            continue
        ax.plot(elapsed, np.degrees(recorded_hand[:, i]), label='Recorded episode', color='black')
        ax.plot(elapsed, np.degrees(log[f'motor_{i}_target_position_rad']), label='Replay command', color='tab:blue')
        ax.plot(elapsed, np.degrees(log[f'motor_{i}_actual_position_rad']), label='Measured motor', color='tab:orange')
        ax.set_title(f'{i}: {labels[i]}')
        ax.set_ylabel('Angle (deg)')
        ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Episode {args.episode}: recorded hand vs commanded and measured replay (episode seconds)')
    fig.tight_layout()
    fig.savefig(args.output / 'hand_comparison.png', dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
