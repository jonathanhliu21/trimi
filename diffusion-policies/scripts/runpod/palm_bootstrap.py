"""Run one palm conditioning experiment; persist artifacts and terminate this pod."""
import hashlib
import json
import os
from pathlib import Path
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
RUN = BASE / 'runs' / MODE
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
    status('verifying', started_timestamp=started)
    threading.Thread(target=heartbeat, daemon=True).start()
    # Independent process survives a killed bootstrap/trainer and enforces a hard ceiling.
    deadline = "import os,time,runpy; time.sleep(float(os.environ.get('MAX_RUNTIME_SECONDS','86400'))); m=runpy.run_path(os.environ['TRIMI_BOOTSTRAP']); m['status']('timed_out'); os.sync(); m['terminate']()"
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
    for archive, destination in [('source.tar.gz', LOCAL / 'code'),
                                 ('dataset.tar', LOCAL / 'datasets'),
                                 ('dinov3.tar.gz', LOCAL)]:
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BASE / 'staging' / archive) as stream:
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
    common = [python, '-u', 'shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py',
              '--config-name=train_unet_image_trimi_hand_cond_dinov3_shortcut_policy',
              'tasks.dataset.dataset_path=' + str(LOCAL / 'datasets' / DATASET_NAME),
              'policy.obs_encoder.vision_backbone.weights=/workspace/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
              'policy.obs_encoder.vision_backbone.repo=' + str(LOCAL / 'dinov3'),
              'policy.obs_encoder.vision_backbone.source=local',
              'training.freeze_encoder=false', 'policy.obs_encoder.vision_backbone.trainable=true',
              '+training.verify_backbone_gradients=true', 'logging.project=trimi-training',
              'training.checkpoint_every=10', f'checkpoint.topk.k={CHECKPOINT_TOPK}',
              'training.seed=42', 'tasks.dataset.seed=42']
    if MODE == 'camera_hand':
        common += ['++tasks.shape_meta.obs.hand_pose_abs={shape:[13],type:low_dim}']
    metadata = dict(campaign=CAMPAIGN, mode=MODE, pod_id=os.environ['RUNPOD_POD_ID'],
                    observations=['camera_0'] + (['hand_pose_abs'] if MODE == 'camera_hand' else []),
                    epochs=500, batch_size=256, seed=42, checkpoint_directory=str(RUN / 'checkpoints'),
                    checkpoint_policy=f'latest plus best {CHECKPOINT_TOPK} by train_loss, every 10 epochs and final',
                    command=common, wandb_project='trimi-training',
                    max_runtime_seconds=float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')))
    (RUN / 'run_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    status('smoke_test')
    command(common + ['hydra.run.dir=' + str(RUN / 'smoke'), 'training.num_epochs=1',
                      'training.max_train_steps=2', 'training.max_val_steps=1',
                      'logging.mode=disabled'], 'smoke.log', 1800)
    if 'BACKBONE_VERIFIED' not in (RUN / 'smoke.log').read_text():
        raise RuntimeError('DINOv3 gradient and weight-update check did not pass')
    reload_code = "import torch,dill; p=torch.load(" + repr(str(RUN / 'smoke/checkpoints/latest.ckpt')) + ",map_location='cpu',pickle_module=dill,weights_only=False); assert 'model' in p['state_dicts']; print('Checkpoint reload verified')"
    command([python, '-c', reload_code], 'reload.log', 300)
    if os.environ.get('TRIMI_KEEP_SMOKE_CHECKPOINTS', '1') == '0':
        # Discard only this run's two-step test weights after the reload succeeds.
        # Full-training checkpoints, resolved test configs, and logs are retained.
        shutil.rmtree(RUN / 'smoke' / 'checkpoints')
        os.sync()
    status('training', backbone_verified=True, checkpoint_reload_verified=True)
    remaining = max(1, int(float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')) - (time.time() - started) - 180))
    command(common + ['hydra.run.dir=' + str(RUN), 'training.num_epochs=500',
                      'logging.mode=online', 'logging.name=' + CAMPAIGN + '_' + MODE,
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
