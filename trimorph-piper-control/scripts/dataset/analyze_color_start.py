"""Offline color-task start analysis. Never opens a hardware connection.

Run from the repository root:
  .venv/bin/python scripts/dataset/analyze_color_start.py
Writes per-episode early samples, statistics, and a plot under output/.
"""
from pathlib import Path
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_collection.replay_actions import Limits, check_joints, jacobian, piper_model


def main():
    dataset = ROOT / 'data_trimorph_color_palm_pick_place'
    out = ROOT / 'output/color_start_analysis'
    out.mkdir(parents=True, exist_ok=True)
    seconds = np.array([0, .1, .25, .5, 1, 2, 3, 5])
    ids, samples, actual_seconds, seeds, calibrations = [], [], [], [], []
    for path in sorted((dataset / 'syncs').glob('*.npz')):
        with np.load(path, allow_pickle=False) as sync:
            assert str(sync['pose_frame']) == 'piper_base'
            arm = np.load(dataset / 'absolute_action_t' / f'{path.stem}.npy')
            hand = np.load(dataset / 'processed_encoder_t' / f'{path.stem}.npy')
            pi, hi = sync['pose_idx'], sync['encoder_idx']
            times = (sync['pose_monotonic_ns'][pi] - sync['pose_monotonic_ns'][pi[0]]) / 1e9
            assert times[-1] >= seconds[-1] and np.all(np.diff(times) > 0)
            indices = np.array([np.argmin(abs(times - t)) for t in seconds])
            values = np.c_[arm[pi[indices]], hand[hi[indices]]]
            assert values.shape == (len(seconds), 19) and np.isfinite(values).all()
            ids.append(path.stem)
            samples.append(values)
            actual_seconds.append(times[indices])
            with (dataset / 'processed' / f'episode_{path.stem}' / 'piper.csv').open() as stream:
                row = next(itertools.islice(csv.DictReader(stream), int(pi[0]), None))
                seeds.append([float(row[f'joint_{j}']) for j in range(1, 7)])
            meta = json.loads((dataset / 'raw' / f'episode_{path.stem}' / 'metadata.json').read_text())
            calibrations.append(meta['trimorph']['hand'])
    samples = np.array(samples, dtype=float)
    starts = samples[:, 0]
    # Each episode has equal weight. Average orientations on SO(3), never Euler angles.
    position = np.median(starts[:, :3], axis=0)
    rotation = R.from_rotvec(starts[:, 3:6]).mean()
    hand_median = np.median(starts[:, 6:], axis=0)
    config = yaml.safe_load((ROOT / 'config/xl330_new_hand_transparent.yaml').read_text())
    for recorded in calibrations:
        for key in ('home_offsets', 'joint_signs'):
            assert all(np.isclose(recorded[key][str(j)], config[key][j]) for j in range(13)), key
    hand = np.clip(hand_median,
                   [config['joint_lower_limits'][j] for j in range(13)],
                   [config['joint_upper_limits'][j] for j in range(13)])
    fk, bounds = piper_model()  # Pure MDH kinematics; no API instance.
    limits = Limits()
    margin = np.radians(limits.joint_margin_deg)
    lower, upper = bounds[:, 0] + margin, bounds[:, 1] - margin
    seed = np.median(seeds, axis=0)
    if seed[4] < 0:
        upper[4] = min(upper[4], -np.radians(limits.wrist_margin_deg))
    else:
        lower[4] = max(lower[4], np.radians(limits.wrist_margin_deg))

    def residual(q):
        pose = fk(q)
        return np.r_[pose[:3] - position, .2 * (R.from_euler('xyz', pose[3:]) * rotation.inv()).as_rotvec()]

    fit = least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                        max_nfev=300, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    q = np.radians(np.round(np.degrees(fit.x), 6))
    assert fit.success and np.linalg.norm(residual(q)) < 1e-6
    check_joints(q, bounds, fk, limits)
    pose = fk(q)
    start_distance = np.linalg.norm(starts[:, :3] - pose[:3], axis=1) * 1000
    start_angle = np.degrees((R.from_rotvec(starts[:, 3:6]) * R.from_euler('xyz', pose[3:]).inv()).magnitude())
    early = []
    for i, t in enumerate(seconds):
        distance = np.linalg.norm(samples[:, i, :3] - starts[:, :3], axis=1) * 1000
        hand_change = np.max(abs(samples[:, i, 6:] - starts[:, 6:]), axis=1)
        early.append(dict(seconds=float(t), xyz_median_mm=(np.median(samples[:, i, :3], axis=0)*1000).tolist(),
                          displacement_mm_p50_p90_max=np.percentile(distance, [50, 90, 100]).tolist(),
                          max_motor_change_rad_p50_p90_max=np.percentile(hand_change, [50, 90, 100]).tolist()))
    report = dict(dataset=str(dataset), episodes=len(ids), name='home_multi_color',
                  method='Median first synchronized XYZ and hand; SO(3) mean orientation; bounded IK seeded by median recorded joints',
                  hardware_access=False, arm_joints_deg=np.degrees(q).tolist(),
                  flange_xyz_m=pose[:3].tolist(), flange_euler_xyz_deg=np.degrees(pose[3:]).tolist(),
                  hand_median_rad=hand_median.tolist(), hand_target_rad=hand.tolist(),
                  hand_clipped_motor_ids=np.flatnonzero(hand != hand_median).tolist(),
                  calibration_hashes=sorted({c['config_sha256'] for c in calibrations}),
                  calibration_offsets_and_signs_match=True,
                  ik_error_mm_deg=[float(np.linalg.norm(residual(q)[:3])*1000), float(np.degrees(np.linalg.norm(residual(q)[3:])/.2))],
                  jacobian_min_singular_value=float(np.linalg.svd(jacobian(fk, q), compute_uv=False)[-1]),
                  distance_to_starts_mm_p50_p90_max=np.percentile(start_distance, [50, 90, 100]).tolist(),
                  angle_to_starts_deg_p50_p90_max=np.percentile(start_angle, [50, 90, 100]).tolist(),
                  early_motion=early,
                  limitation='Dataset-derived configuration, not hardware or collision validated; different object layouts still require visual assessment.')
    (out / 'statistics.json').write_text(json.dumps(report, indent=2) + '\n')
    with (out / 'episode_early_samples.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['episode', 'requested_seconds', 'sample_seconds', 'x_m', 'y_m', 'z_m',
                         'rx_rotvec_rad', 'ry_rotvec_rad', 'rz_rotvec_rad', *[f'hand_{j}_rad' for j in range(13)]])
        for k, episode in enumerate(ids):
            for i, t in enumerate(seconds):
                writer.writerow([episode, t, actual_seconds[k][i], *samples[k, i]])
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/trimorph-color-start-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for ax, (a, b) in zip(axes[0], [(0, 1), (0, 2), (1, 2)]):
        for values in samples:
            ax.plot(values[:5, a]*1000, values[:5, b]*1000, color='gray', alpha=.2)
        ax.scatter(starts[:, a]*1000, starts[:, b]*1000, s=10, label='151 episode starts')
        ax.scatter(pose[a]*1000, pose[b]*1000, color='red', marker='*', s=170, label='home_multi_color')
        ax.set(xlabel=f'{"XYZ"[a]} (mm)', ylabel=f'{"XYZ"[b]} (mm)')
        ax.set_aspect('equal', adjustable='datalim')
    axes[0, 0].legend(fontsize=8)
    for j, ax in enumerate(axes[1]):
        delta = (samples[:, :, j] - starts[:, None, j])*1000
        lo, med, hi = np.percentile(delta, [10, 50, 90], axis=0)
        ax.fill_between(seconds, lo, hi, alpha=.2, label='Episode 10–90%')
        ax.plot(seconds, med, label='Episode median')
        ax.set(xlabel='Seconds from first synchronized pose', ylabel=f'{"XYZ"[j]} displacement (mm)')
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Color palm pick/place: starts and early motion (offline recordings only)')
    fig.savefig(out / 'starts_and_early_motion.png', dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
