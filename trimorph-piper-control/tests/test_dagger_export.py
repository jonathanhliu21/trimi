"""Real-video raw-to-compact export, alignment and destructive-mode rollback."""
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from data_collection import postprocess_dataset as post
from scripts.dataset import postprocess_dataset as base
from scripts.dagger import export_segments as exporter


class ExportTests(unittest.TestCase):
    def write_csv(self, path, rows):
        exporter.write_csv(path, rows)

    def fixture(self, root, index=2, outcome='success', intervals=((30, 75), (85, 110)), bad_hand=False):
        raw = root / 'raw' / f'episode_{index:06d}'
        raw.mkdir(parents=True)
        origin = 1_000_000_000
        frame_ns = np.array([origin + round(i * 1e9 / 30) for i in range(120)], dtype=np.int64)
        meta = dict(status='complete', piper={}, camera={'fps': 30}, dagger=dict(outcome=outcome,
            takeover='kinesthetic', round=1, policy={'sha256': 'abc'},
            human_intervals=[[int(frame_ns[a]), int(frame_ns[b])] for a, b in intervals]))
        (raw / 'metadata.json').write_text(json.dumps(meta))
        video = cv2.VideoWriter(str(raw / 'camera.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30., (32, 24))
        self.assertTrue(video.isOpened())
        for i in range(120):
            video.write(np.full((24, 32, 3), i*2, dtype=np.uint8))
        video.release()
        self.write_csv(raw / 'camera_timestamps.csv', [dict(frame_idx=i, monotonic_ns=int(t)) for i, t in enumerate(frame_ns)])
        stamps = [origin+i*10_000_000 for i in range(401)]
        self.write_csv(raw / 'leader_hand.csv', [dict(monotonic_ns=t+(10_000_000_000 if bad_hand else 0),
            **{f'pos_{j}': i*.001+j*.01 for j in range(13)}) for i, t in enumerate(stamps)])
        self.write_csv(raw / 'piper.csv', [dict(monotonic_ns=t, x=i*.0001, y=-.2, z=.3, roll=.1, pitch=.2, yaw=.3)
                                         for i, t in enumerate(stamps)])
        return raw, frame_ns

    def snapshot(self, root):
        return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in root.rglob('*') if p.is_file()}

    def run_export(self, *args, **kwargs):
        kwargs.setdefault('trim', True)
        with contextlib.redirect_stdout(io.StringIO()):
            return exporter.export(*args, **kwargs)

    def test_default_keeps_entire_video_and_syncs_once_per_rollout(self):
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/'dataset', Path(d)/'full'
            raw, _ = self.fixture(root)
            self.fixture(root, 3, outcome='failure')
            self.fixture(root, 4, intervals=())
            original_video = (raw/'camera.mp4').read_bytes()
            original_intervals = json.loads((raw/'metadata.json').read_text())['dagger']['human_intervals']
            with contextlib.redirect_stdout(io.StringIO()), patch.object(exporter, 'trim_video', side_effect=AssertionError('must not trim')):
                records, skipped = exporter.export(root, out, min_rows=5)
            self.assertEqual(len(records), 1)  # two human intervals, one whole rollout
            self.assertFalse(records[0]['trimmed'])
            paths = base.processed_paths(out, 0)
            self.assertEqual(paths.video_path.read_bytes(), original_video)
            with np.load(paths.sync_path) as sync:
                np.testing.assert_array_equal(sync['frame_idx'], np.arange(120))
                self.assertEqual(str(sync['source']), 'dagger_rollout')
                self.assertFalse(sync['intervention'][0])
                self.assertTrue(sync['intervention'][30])
                self.assertTrue(sync['intervention'][100])
                self.assertEqual(len(sync['encoder_idx']), 120)
            self.assertEqual(len(np.load(paths.hand_path)), 401)
            saved = json.loads((out/'raw'/'episode_000000'/'metadata.json').read_text())
            self.assertEqual(saved['dagger']['human_intervals'], original_intervals)
            self.assertEqual(len(skipped), 2)
            # A full export can subsequently be trimmed using the retained intervals.
            compact = Path(d)/'compact'
            trimmed, _ = self.run_export(out, compact, min_rows=5)
            self.assertEqual([r['rows'] for r in trimmed], [46, 26])

    def test_default_in_place_prunes_but_keeps_full_retained_rollout(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'dataset'
            raw, _ = self.fixture(root)
            self.fixture(root, 3, outcome='failure')
            self.fixture(root, 4, intervals=())
            original_video = (raw/'camera.mp4').read_bytes()
            self.run_export(root, overwrite_raw=True, trim=False, min_rows=5)
            self.assertEqual([p.name for p in (root/'raw').iterdir()], ['episode_000000'])
            self.assertEqual((root/'raw'/'episode_000000'/'camera.mp4').read_bytes(), original_video)
            with np.load(base.processed_paths(root,0).sync_path) as sync:
                np.testing.assert_array_equal(sync['frame_idx'], np.arange(120))

    def test_cli_trim_is_opt_in(self):
        self.assertFalse(exporter.parse_args(['root', '--out', 'out']).trim)
        self.assertTrue(exporter.parse_args(['root', '--out', 'out', '--trim']).trim)

    def test_raw_selection_compaction_and_roundtrip_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/'rawdata', Path(d)/'human'
            raw, stamps = self.fixture(root)
            # Invalid assets in rejected episodes prove selection happens first.
            failure, _ = self.fixture(root, 3, outcome='failure')
            (failure/'camera.mp4').write_bytes(b'not a video')
            self.fixture(root, 4, intervals=())
            before = self.snapshot(root)
            with patch.object(post, 'process_episode', wraps=post.process_episode) as process:
                records, skipped = self.run_export(root, out, min_rows=5)
            self.assertEqual(process.call_count, 1)
            self.assertEqual(before, self.snapshot(root))
            self.assertEqual([r['rows'] for r in records], [46, 26])
            self.assertEqual([r['rollout'] for r in records], [2, 2])
            self.assertEqual(len(skipped), 2)
            for number, record in enumerate(records):
                paths = base.processed_paths(out, number)
                with np.load(paths.sync_path, allow_pickle=False) as sync:
                    n = record['rows']
                    np.testing.assert_array_equal(sync['frame_idx'], np.arange(n))
                    self.assertEqual(str(sync['source']), 'dagger_human')
                    self.assertEqual(int(sync['encoder_idx'][0]), 0)
                    self.assertEqual(int(sync['pose_idx'][0]), 0)
                    expected_hand = sync['encoder_monotonic_ns'] / 1e9 - 1
                    np.testing.assert_allclose(np.load(paths.hand_path)[:, 0], expected_hand*.1, atol=1e-6)
                    cap = cv2.VideoCapture(str(paths.video_path))
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n)
                    ok, first = cap.read()
                    cap.release()
                    self.assertTrue(ok)
                    self.assertLess(abs(float(first.mean()) - 2*record['original_frame_range'][0]), 10)
                    raw_video = out/'raw'/f'episode_{number:06d}'/'camera.mp4'
                    self.assertEqual(os.stat(raw_video).st_ino, os.stat(paths.video_path).st_ino)
            # Shared postprocessor can regenerate equivalent training data from compact raw.
            config = post.PostprocessConfig(dataset_root=out, raw_dir=out/'raw', overwrite_processed_files=True,
                process_only_raw_statuses={'complete'}, max_sync_error_s=.01, trim_camera_frames=0, pose_source='piper')
            expected = [(np.load(base.processed_paths(out, i).hand_path), np.load(base.processed_paths(out, i).eef_path)) for i in range(2)]
            with contextlib.redirect_stdout(io.StringIO()):
                for episode in base.discover_raw_episodes(config):
                    post.process_episode(config, episode)
            for i, (hand, poses) in enumerate(expected):
                np.testing.assert_allclose(np.load(base.processed_paths(out,i).hand_path), hand)
                np.testing.assert_allclose(np.load(base.processed_paths(out,i).eef_path), poses)

    def test_destination_required_and_existing_files_never_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/'rawdata', Path(d)/'out'
            self.fixture(root)
            for args in ([str(root)], [str(root), '--out', str(out), '--overwrite-raw']):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    exporter.parse_args(args)
            for kwargs in ({}, {'out': out, 'overwrite_raw': True}, {'out': root/'nested'}, {'out': root.parent}):
                with self.assertRaises(ValueError):
                    self.run_export(root, **kwargs)
            out.mkdir()
            (out/'keep.txt').write_text('untouched')
            before = self.snapshot(root.parent)
            with self.assertRaises(FileExistsError):
                self.run_export(root, out)
            self.assertEqual(before, self.snapshot(root.parent))

    def test_empty_destination_and_source_changes_during_export(self):
        with tempfile.TemporaryDirectory() as d:
            root, out = Path(d)/'dataset', Path(d)/'out'
            raw, _ = self.fixture(root)
            out.mkdir()
            self.run_export(root, out, min_rows=5)
            self.assertTrue((out/'segments.json').exists())
            original_process = post.process_episode

            def collection_appended(config, episode):
                value = original_process(config, episode)
                (raw/'new-data.csv').write_text('collection is active')
                return value

            video_before = (raw/'camera.mp4').read_bytes()
            with patch.object(post, 'process_episode', side_effect=collection_appended), self.assertRaisesRegex(RuntimeError, 'changed during export'):
                self.run_export(root, overwrite_raw=True, min_rows=5)
            self.assertEqual((raw/'camera.mp4').read_bytes(), video_before)
            self.assertFalse((root/'segments.json').exists())

    def test_in_place_removes_rejected_raw_and_stale_processed_data(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'dataset'
            self.fixture(root)
            self.fixture(root, 3, outcome='failure')
            self.fixture(root, 4, intervals=())
            base.ensure_processed_dirs(root)
            (root/'syncs'/'999999.npz').write_bytes(b'stale')
            (root/'midas_test.zarr.zip').write_bytes(b'cached full rollouts')
            (root/'notes.txt').write_text('preserve')
            self.run_export(root, overwrite_raw=True, min_rows=5)
            self.assertEqual(sorted(p.name for p in (root/'raw').iterdir()), ['episode_000000','episode_000001'])
            self.assertFalse((root/'syncs'/'999999.npz').exists())
            self.assertFalse((root/'midas_test.zarr.zip').exists())
            self.assertEqual((root/'notes.txt').read_text(), 'preserve')
            self.assertEqual(len(list((root/'camera'/'mp4_files').glob('*.mp4'))), 2)
            self.assertFalse(list(root.parent.glob('.*export*')))

    def test_timing_misalignment_warns_and_exports_both_modes(self):
        for trim in (False, True):
            with self.subTest(trim=trim), tempfile.TemporaryDirectory() as d:
                root, out = Path(d)/'dataset', Path(d)/'out'
                raw, _ = self.fixture(root)
                rows = base.read_csv_rows(raw/'leader_hand.csv')
                # A recording gap during the correction forces nearest samples >10 ms away.
                rows = [r for r in rows if not 1_900_000_000 < int(r['monotonic_ns']) < 2_100_000_000]
                self.write_csv(raw/'leader_hand.csv', rows)
                before = self.snapshot(root)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    segments, _ = exporter.export(root, out, trim=trim, min_rows=5)
                self.assertIn('keeping best-effort sync', output.getvalue())
                self.assertTrue(any(s['max_sync_error_ms']['encoder_idx'] > 10 for s in segments))
                self.assertEqual(before, self.snapshot(root))

    def test_invalid_video_never_changes_originals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'dataset'
            raw, _ = self.fixture(root)
            (raw/'camera.mp4').write_bytes(b'broken video')
            before = self.snapshot(root)
            with self.assertRaises(ValueError):
                self.run_export(root, overwrite_raw=True, min_rows=5)
            self.assertEqual(before, self.snapshot(root))

    def test_publication_failure_rolls_back_originals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'dataset'
            self.fixture(root)
            self.fixture(root, 3, outcome='failure')
            before = self.snapshot(root)
            real_replace = os.replace
            failed = False
            def fail_once(src, dst):
                nonlocal failed
                if not failed and Path(dst) == root/'syncs' and 'result' in Path(src).parts:
                    failed = True
                    raise OSError('injected publication failure')
                return real_replace(src, dst)
            with patch.object(exporter.os, 'replace', side_effect=fail_once), self.assertRaisesRegex(OSError, 'injected'):
                self.run_export(root, overwrite_raw=True, min_rows=5)
            self.assertEqual(before, self.snapshot(root))

    def test_empty_or_short_selection_does_not_delete_anything(self):
        for intervals, outcome in (((), 'success'), (((30,31),), 'success'), (((30,75),), 'failure')):
            with self.subTest(intervals=intervals,outcome=outcome), tempfile.TemporaryDirectory() as d:
                root = Path(d)/'dataset'
                self.fixture(root, outcome=outcome, intervals=intervals)
                before = self.snapshot(root)
                with self.assertRaisesRegex(ValueError, 'nothing changed'):
                    self.run_export(root, overwrite_raw=True)
                self.assertEqual(before,self.snapshot(root))


if __name__ == '__main__':
    unittest.main()
