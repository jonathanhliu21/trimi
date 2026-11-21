"""Unattended training: smoke test, full run, durable status, terminate this Pod."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path('/workspace/code')
RUN = Path('/workspace/runs/m3_screw_shortcut_dinov3_001')
STATE = RUN / 'status.json'
BASE = 'https://api.runpod.io/v2'


def status(phase, **fields):
    RUN.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(dict(phase=phase, timestamp=time.time(), pod_id=os.environ['RUNPOD_POD_ID'], **fields), stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(STATE)
    os.sync()


def terminate():
    # This path never deletes a network volume, only the current Pod.
    url = BASE + '/pods/' + os.environ['RUNPOD_POD_ID']
    while True:
        request = urllib.request.Request(url, method='DELETE', headers={
            'Authorization': 'Bearer ' + os.environ['RUNPOD_SHUTDOWN_KEY']})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status in (200, 202, 204):
                    print('Pod termination accepted', flush=True)
                    return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return
            print('Pod termination retry: HTTP', error.code, flush=True)
        except Exception as error:
            print('Pod termination retry:', type(error).__name__, flush=True)
        time.sleep(10)


def command(argv, log, timeout):
    with log.open('w') as stream:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if result:
        raise RuntimeError(f'{log.name} exited with status {result}')


def main():
    started = time.time()
    status('setup')
    # Separate watchdog process survives failure or SIGKILL of this runner.
    subprocess.Popen([sys.executable, str(Path(__file__).with_name('deadline.py'))],
                     start_new_session=True, stdin=subprocess.DEVNULL)
    try:
        if not os.path.ismount('/workspace'):
            raise RuntimeError('/workspace must be the persistent network-volume mount')
        command([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-dinov3.txt',
                 'python-dotenv'], RUN / 'install.log', 1800)
        os.environ['PYTHONUNBUFFERED'] = '1'
        os.environ['WANDB_ENTITY'] = 'jonathanhliu21-ucla'
        os.environ['WANDB_INIT_TIMEOUT'] = '120'
        common = [sys.executable, '-u', 'shortcut_policy/unet_shortcut/train_unet_image_shortcut_policy.py',
                  '--config-name=train_unet_image_trimi_hand_cond_dinov3_shortcut_policy',
                  'tasks.dataset.dataset_path=/workspace/datasets/data_trimorph_m3_screw_pick_place',
                  'policy.obs_encoder.vision_backbone.weights=/workspace/models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
                  'training.freeze_encoder=false', 'policy.obs_encoder.vision_backbone.trainable=true',
                  '+training.verify_backbone_gradients=true', 'logging.project=trimi-training',
                  'training.checkpoint_every=10']
        status('smoke_test')
        command(common + ['hydra.run.dir=' + str(RUN / 'smoke'), 'training.num_epochs=1',
                          'training.max_train_steps=2', 'training.max_val_steps=1',
                          'logging.mode=disabled'], RUN / 'smoke.log', 1800)
        if 'BACKBONE_VERIFIED' not in (RUN / 'smoke.log').read_text():
            raise RuntimeError('Backbone gradient/update check did not complete')
        # Exercise checkpoint deserialization on the actual trained model.
        command([sys.executable, '-c',
                 "import torch,dill; p=torch.load('/workspace/runs/m3_screw_shortcut_dinov3_001/smoke/checkpoints/latest.ckpt',map_location='cpu',pickle_module=dill,weights_only=False); assert 'model' in p['state_dicts']; print('Checkpoint reload verified')"],
                RUN / 'reload.log', 300)
        if os.environ.get('TRIMI_SMOKE_ONLY') == '1':
            status('smoke_completed', backbone_verified=True, checkpoint_reload_verified=True)
            return
        status('training', backbone_verified=True)
        remaining = max(1, int(float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')) - (time.time() - started) - 180))
        command(common + ['hydra.run.dir=' + str(RUN), 'training.num_epochs=500',
                          'logging.mode=online', 'logging.name=m3_screw_shortcut_dinov3_001'],
                RUN / 'train.log', remaining)
        checkpoint = RUN / 'checkpoints/latest.ckpt'
        digest = hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        (RUN / 'checkpoint.sha256').write_text(digest.hexdigest() + '  checkpoints/latest.ckpt\n')
        status('completed', checkpoint=str(checkpoint), sha256=digest.hexdigest())
    except BaseException as error:
        status('failed', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        os.sync()
        terminate()


if __name__ == '__main__':
    main()
