"""Run one palm conditioning experiment; persist artifacts and terminate this pod."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.request

CAMPAIGN = os.environ.get('TRIMI_CAMPAIGN', 'palm_multi_dinov3_20261123_01')
if not CAMPAIGN or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in CAMPAIGN):
    raise ValueError('TRIMI_CAMPAIGN must be a simple experiment directory name')
DATASET_NAME = os.environ.get('TRIMI_DATASET_NAME', 'data_trimorph_palm_pick_place_multi')
if not DATASET_NAME or Path(DATASET_NAME).name != DATASET_NAME:
    raise ValueError('TRIMI_DATASET_NAME must be a dataset directory name')
CHECKPOINT_TOPK = int(os.environ.get('TRIMI_CHECKPOINT_TOPK', '2'))
if CHECKPOINT_TOPK < 0:
    raise ValueError('TRIMI_CHECKPOINT_TOPK must be nonnegative')
BASE = Path('/workspace/experiments') / CAMPAIGN
MODE = os.environ.get('TRIMI_OBSERVATION_MODE', '')
VARIANT = os.environ['TRIMI_VARIANT']
RUN = BASE / 'runs' / VARIANT
LOCAL = Path('/tmp/palm_training')
STATE = {}
LOCK = threading.Lock()


def terminate():
    """Delete only this pod. Network-volume data is retained."""
    url = 'https://api.runpod.io/v2/pods/' + os.environ['RUNPOD_POD_ID']
    while True:
        request = urllib.request.Request(url, method='DELETE', headers={
            'Authorization': 'Bearer ' + os.environ['RUNPOD_SHUTDOWN_KEY']})
        try:
            with urllib.request.urlopen(request, timeout=30):
                print('Pod termination accepted', flush=True)
                return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return
            print('Termination retry HTTP', error.code, flush=True)
        except Exception as error:
            print('Termination retry', type(error).__name__, flush=True)
        time.sleep(10)


def status(phase=None, **fields):
    with LOCK:
        if phase is not None:
            STATE['phase'] = phase
        STATE.update(fields, heartbeat=time.time(), pod_id=os.environ['RUNPOD_POD_ID'], mode=MODE)
        RUN.mkdir(parents=True, exist_ok=True)
        temporary = RUN / 'status.json.tmp'
        with temporary.open('w') as stream:
            json.dump(STATE, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(RUN / 'status.json')


def heartbeat():
    while True:
        time.sleep(30)
        status()


def command(argv, log, timeout, cwd=LOCAL / 'code'):
    with (RUN / log).open('w') as stream:
        process = subprocess.Popen(argv, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if result:
        raise RuntimeError(f'{log} exited with status {result}')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if MODE not in ('camera', 'camera_hand'):
        raise ValueError('TRIMI_OBSERVATION_MODE must be camera or camera_hand')
    if not os.path.ismount('/workspace'):
        raise RuntimeError('Persistent network volume is not mounted at /workspace')
    RUN.mkdir(parents=True, exist_ok=True)
    # Bootstrap logs remain on the volume after this container is deleted.
    log = (RUN / 'bootstrap.log').open('a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    started = time.time()
    if os.environ.get('TRIMI_RUNTIME_FROM_START') == '1':
        os.environ['TRIMI_DEADLINE_TIMESTAMP'] = str(min(
            float(os.environ['TRIMI_DEADLINE_TIMESTAMP']),
            started + float(os.environ.get('MAX_RUNTIME_SECONDS', '86400'))))
    status('verifying', started_timestamp=started)
    threading.Thread(target=heartbeat, daemon=True).start()
    # Independent process survives a killed bootstrap/trainer and enforces a hard ceiling.
    deadline = """
import os,time,runpy
parent = os.getppid()
cutoff = float(os.environ['TRIMI_DEADLINE_TIMESTAMP'])
while time.time() < cutoff:
    try:
        os.kill(parent, 0)
    except ProcessLookupError:
        break
    time.sleep(10)
m = runpy.run_path(os.environ['TRIMI_BOOTSTRAP'])
m['status']('timed_out' if time.time() >= cutoff else 'failed', error='Independent watchdog fired')
os.sync()
m['terminate']()
"""
    subprocess.Popen([sys.executable, '-u', '-c', deadline], start_new_session=True,
                     stdin=subprocess.DEVNULL)
    manifest = json.loads((BASE / 'staging/manifest.json').read_text())
    for key, expected in manifest['credential_sha256'].items():
        if hashlib.sha256(os.environ.get(key, '').encode()).hexdigest() != expected:
            raise RuntimeError(f'{key} does not match the authorized local .env credential')
    for artifact in manifest['artifacts']:
        path = Path('/workspace') / artifact['key']
        if path.stat().st_size != artifact['size'] or digest(path) != artifact['sha256']:
            raise RuntimeError('Artifact checksum mismatch: ' + artifact['key'])
        print('Checksum verified:', artifact['key'], flush=True)
    free_gib = os.statvfs('/workspace').f_bavail * os.statvfs('/workspace').f_frsize / 1024**3
    minimum_free_gib = float(os.environ.get('TRIMI_MIN_FREE_GIB', '12'))
    print(f'Checkpoint volume free space: {free_gib:.2f} GiB', flush=True)
    if free_gib < minimum_free_gib:
        raise RuntimeError(f'Less than {minimum_free_gib:g} GiB free on checkpoint volume')
    status('extracting')
    LOCAL.mkdir(parents=True, exist_ok=True)
    # A dataset may be staged as one dataset.tar or as dataset.partNN.tar pieces.
    for pattern, destination in [(r'source\.tar\.gz', LOCAL / 'code'),
                                 (r'dataset(\.part\d+)?\.tar', LOCAL / 'datasets'),
                                 (r'dinov3\.tar\.gz', LOCAL)]:
        destination.mkdir(parents=True, exist_ok=True)
        paths = [Path('/workspace') / a['key'] for a in manifest['artifacts']
                 if re.fullmatch(pattern, Path(a['key']).name)]
        if not paths:
            raise RuntimeError('No staged archive matches ' + pattern)
        for archive_path in paths:
            with tarfile.open(archive_path) as stream:
                stream.extractall(destination, filter='data')
    # Each pod owns its environment and caches, avoiding concurrent writes to shared code.
    os.environ.update(PYTHONUNBUFFERED='1', WANDB_ENTITY='jonathanhliu21-ucla',
                      WANDB_INIT_TIMEOUT='120', WANDB_CONSOLE='off',
                      WANDB_DIR=str(RUN), WANDB_CACHE_DIR='/tmp/wandb-cache',
                      MPLCONFIGDIR='/tmp/matplotlib', TORCH_HOME='/tmp/torch-cache',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    environment = Path('/opt/palm-venv')
    status('installing')
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
    python = str(environment / 'bin/python')
    command([python, '-m', 'pip', 'install', '--no-cache-dir', '-r', 'requirements-dinov3.txt',
             'python-dotenv'], 'install.log', 1800)
    gpu_check = """
import torch,json,os
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
p = torch.cuda.get_device_properties(0)
assert os.environ.get('TRIMI_EXPECTED_GPU_FAMILY', 'RTX PRO 6000') in p.name
assert p.total_memory > float(os.environ.get('TRIMI_MIN_GPU_MEMORY_GIB', '90')) * 1024**3
print(json.dumps(dict(name=p.name, memory_bytes=p.total_memory, torch=torch.__version__, cuda=torch.version.cuda)))
"""
    command([python, '-c', gpu_check], 'gpu_verification.json', 120)
    common = [python, '-u', 'shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py',
              '--config-name=train_unet_image_trimi_hand_cond_dinov3_shortcut_policy',
              'tasks.dataset.dataset_path=' + str(LOCAL / 'datasets' / DATASET_NAME),
              'policy.obs_encoder.vision_backbone.weights=/workspace/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
              'policy.obs_encoder.vision_backbone.repo=' + str(LOCAL / 'dinov3'),
              'policy.obs_encoder.vision_backbone.source=local',
              'training.freeze_encoder=false', 'policy.obs_encoder.vision_backbone.trainable=true',
              '+training.verify_backbone_gradients=true', '+training.verify_experiment=true',
              'tasks.dataset.max_train_episodes=' + os.environ.get('TRIMI_MAX_TRAIN_EPISODES', 'null'), 'logging.project=trimi-training',
              'training.checkpoint_every=50', f'checkpoint.topk.k={CHECKPOINT_TOPK}',
              'training.seed=42', 'tasks.dataset.seed=42']
    if MODE == 'camera_hand':
        common += ['++tasks.shape_meta.obs.hand_pose_abs={shape:[13],type:low_dim}']
    metadata = dict(campaign=CAMPAIGN, mode=MODE, pod_id=os.environ['RUNPOD_POD_ID'],
                    observations=['camera_0'] + (['hand_pose_abs'] if MODE == 'camera_hand' else []),
                    epochs=500, batch_size=256, seed=42, checkpoint_directory=str(RUN / 'checkpoints'),
                    max_train_episodes=os.environ.get('TRIMI_MAX_TRAIN_EPISODES', 'null'),
                    checkpoint_policy=f'latest plus best {CHECKPOINT_TOPK} by train_loss, every 50 epochs and final',
                    command=common, wandb_project='trimi-training',
                    max_runtime_seconds=float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')))
    (RUN / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    status('smoke_test')
    command(common + ['hydra.run.dir=' + str(RUN / 'smoke'), 'training.num_epochs=1',
                      'training.max_train_steps=2', 'training.max_val_steps=1',
                      'logging.mode=disabled'], 'smoke.log', 1800)
    if 'BACKBONE_VERIFIED' not in (RUN / 'smoke.log').read_text():
        raise RuntimeError('DINOv3 gradient and weight-update check did not pass')
    reload_code = """
import torch,dill,hydra
from shortcut_policy.unet_shortcut.train_unet_image_shortcut_policy import TrainDiffusionUnetImageWorkspace
payload=torch.load(CHECKPOINT,map_location='cpu',pickle_module=dill,weights_only=False)
cfg=payload['cfg']
cfg.policy.obs_encoder.vision_backbone.pretrained=False
workspace=TrainDiffusionUnetImageWorkspace(cfg, output_dir='/tmp/checkpoint-reload-verification')
workspace.load_payload(payload)
for name, value in payload['state_dicts']['model'].items():
    assert torch.equal(workspace.model.state_dict()[name].cpu(), value), name
print('Checkpoint model, EMA and optimizer reload verified')
""".replace('CHECKPOINT', repr(str(RUN / 'smoke/checkpoints/latest.ckpt')))
    command([python, '-c', reload_code], 'reload.log', 300)
    if os.environ.get('TRIMI_KEEP_SMOKE_CHECKPOINTS', '1') == '0':
        # Discard only this run's two-step test weights after the reload succeeds.
        # Full-training checkpoints, resolved test configs, and logs are retained.
        shutil.rmtree(RUN / 'smoke' / 'checkpoints')
        os.sync()
    split = json.loads((RUN / 'smoke/split_verification.json').read_text())
    expected = json.loads((BASE / 'staging/dataset_report.json').read_text())
    assert split['episodes'] == expected['n_episodes']
    assert split['validation'] == expected['validation_indices']
    key = 'train_100_indices' if os.environ.get('TRIMI_MAX_TRAIN_EPISODES') == '100' else 'train_all_indices'
    assert split['train'] == expected[key]
    expected_observations = {'camera_0'} | ({'hand_pose_abs'} if MODE == 'camera_hand' else set())
    assert set(split['observations']) == expected_observations
    if MODE == 'camera_hand':
        assert split['observations']['hand_pose_abs'][-1] == 13
    status('training', backbone_verified=True, checkpoint_reload_verified=True, observations_verified=True, gpu_verified=True,
           train_episode_count=len(split['train']), validation_episode_count=len(split['validation']))
    remaining = max(1, int(float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')) - (time.time() - started) - 180))
    command(common + ['hydra.run.dir=' + str(RUN), 'training.num_epochs=500',
                      'logging.mode=online', 'logging.name=' + CAMPAIGN + '_' + VARIANT,
                      'logging.id=' + os.environ['TRIMI_WANDB_RUN_ID']], 'train.log', remaining)
    checkpoint = RUN / 'checkpoints/latest.ckpt'
    checksum = digest(checkpoint)
    (RUN / 'checkpoint.sha256').write_text(checksum + '  checkpoints/latest.ckpt\n')
    status('completed', checkpoint=str(checkpoint), sha256=checksum)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('Training failed:', type(error).__name__, str(error), flush=True)
        try:
            status('failed', error_type=type(error).__name__, error=str(error))
        except Exception:
            pass
    finally:
        os.sync()
        terminate()
